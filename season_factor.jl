include("kl_optim.jl")
include("local_level_vb.jl")
include("linear_growth.jl")

begin
	using Distributions, Random
	using LinearAlgebra
	using StatsFuns
	using MCMCChains
	using Statistics
	using DataFrames
	using StatsPlots
	using SpecialFunctions
	using StatsBase
	using Dates
	using DataFrames
end

function gen_sea(A, C, R, Q, T, x_1)
    K = size(A, 1)
    D = size(R, 1)
    x = zeros(K, T)
    y = zeros(D, T)

    x[:, 1] = x_1  

    if D == 1
        y[:, 1] = C * x[:, 1] + rand(MvNormal(zeros(D), sqrt.(R)))
    else
        y[:, 1] = C * x[:, 1] + rand(MvNormal(zeros(D), R))
    end

    for t in 2:T
        x[:, t] = A * x[:, t-1] + rand(MvNormal(zeros(K), Q))

        if D == 1
            y[:, t] = C * x[:, t] + rand(MvNormal(zeros(D), sqrt.(R))) # linear growth 
        else
            y[:, t] = C * x[:, t] + rand(MvNormal(zeros(D), R)) 
        end
    end

    return y, x
end

function vb_m_static(y, S_C, W_C, hpp::HPP_D, C::Array{Float64, 2})
    D, T = size(y)
	
	G = y*y' - 2 * C * S_C + C * W_C * C'
    a_ = hpp.a + 0.5 * T
	a_s = a_ * ones(D)
    b_s = [hpp.b + 0.5 * G[i, i] for i in 1:D]
	q_ρ = Gamma.(a_s, 1 ./ b_s)
	Exp_R⁻¹ = diagm(mean.(q_ρ))
    return Exp_R⁻¹, a_, b_s
end

function test_m_static()
    T = 500  # Number of time steps
    rho = 0.0  # System evolution parameter, static seasonal model: rho = 0
    r = 0.1  # Observation noise variance
    A = [-1.0 -1.0 -1.0; 1.0 0.0 0.0; 0.0 1.0 0.0]  # State transition matrix
    C = [1.0 0.0 0.0]  # Emission matrix
    Q = Diagonal([rho, 0.0, 0.0])  # System evolution noise covariance
    R = [r]

    x_1 = [0.50, 0.35, 0.15] # Satisfies identifiability constraint: sum to 0
	Random.seed!(123)
    y, x = gen_sea(A, C, R, Q, T, x_1)

	W_C = sum(x[:, t] * x[:, t]' for t in 1:T)
	S_C = sum(x[:, t] * y[:, t]' for t in 1:T)

	prior = HPP_D(0.01, 0.01, 0.01, 0.01, zeros(3), Matrix{Float64}(I, 3, 3))
    vb_m_static(y, S_C, W_C, prior, C)
end

#test_m_static()

function vb_e_static(y, A, C, E_R, prior::HPP_D, smooth_out = false)
    P, T = size(y)  # Number of time points
    K = size(A, 1)  # Dimension of the latent state

    μ_0, Σ_0 = prior.μ_0, prior.Σ_0
    
    # Initialize the filtered means and covariances
    μ_f = zeros(K, T)
    Σ_f = zeros(K, K, T)
    f_s = zeros(P, T)
	S_s = zeros(P, P, T)
	
    # Set the initial filtered mean and covariance to their prior values
	A_1 = A * μ_0
	R_1 = A * Σ_0 * A'
	
	f_s[:, 1] = f_1 = C * A_1
	S_s[:, :, 1] = S_1 = C * R_1 * C' + E_R
	
    μ_f[:, 1] = A_1 + R_1 * C'* inv(S_1) * (y[:, 1] - f_1)
    Σ_f[:, :, 1] = R_1 - R_1*C'*inv(S_1)*C*R_1 
    
    # Forward pass (kalman filter)
    for t in 2:T
        # Predicted state mean and covariance
        μ_p = A * μ_f[:, t-1]
        Σ_p = A * Σ_f[:, :, t-1] * A'

		# marginal y - normalization
		f_s[:, t] = f_t = C * μ_p
		S_s[:, :, t] = S_t = C * Σ_p * C' + E_R
		
		# Filtered state mean and covariance (2.8a - 2.8c DLM with R)
		μ_f[:, t] = μ_p + Σ_p * C' * inv(S_t) * (y[:, t] - f_t)
		Σ_f[:, :, t] = Σ_p - Σ_p * C' * inv(S_t) * C * Σ_p	
    end

    log_z = sum(logpdf(MvNormal(f_s[:, i], Symmetric(S_s[:, :, i])), y[:, i]) for i in 1:T)

    if (smooth_out)
		return μ_f, Σ_f
	end

    W_C = sum(Σ_f[:, :, t] + μ_f[:, t] * μ_f[:, t]' for t in 1:T)
	S_C = sum(μ_f[:, t] * y[:, t]' for t in 1:T)

    return W_C, S_C, log_z
end

function test_e_static()
    T = 500  # Number of time steps
    rho = 0.0  # System evolution parameter, static seasonal model: rho = 0
    r = 0.1  # Observation noise variance
    A = [-1.0 -1.0 -1.0; 1.0 0.0 0.0; 0.0 1.0 0.0]  # State transition matrix
    C = [1.0 0.0 0.0]  # Emission matrix
    Q = Diagonal([rho, 0.0, 0.0])  # System evolution noise covariance
    R = [r]

    x_1 = [0.50, 0.35, 0.15] # Satisfies identifiability constraint: sum to 0
	Random.seed!(123)
    y, x = gen_sea(A, C, R, Q, T, x_1)

    μs, _ = vb_e_static(y, A, C, R, true)

    println(error_metrics(x, μs))
    plot_latent(x', μs')
end

#test_e_static()

function vb_ss(ys, A, C, prior::HPP_D, max_iter=500, tol=1e-4, init="random")
	D, _ = size(ys)
    E_R = missing
    W_C, S_C = missing, missing

    if init == "random"
        r = rand(InverseGamma(prior.a, prior.b))
        R_init = diagm(ones(D) .* r)
        W_C, S_C, _ = vb_e_static(ys, A, C, R_init, prior)
    end

    if init == "fixed"
        W_C, S_C = ones(size(A)), ones(size(C'))
    end

    elbo_prev = -Inf
	el_s = zeros(max_iter)

    for i in 1:max_iter
		E_R, a_q, b_q = vb_m_static(ys, S_C, W_C, prior, C)
		W_C, S_C, log_z = vb_e_static(ys, A, C, inv(E_R), prior)

        kl_ρ = sum([kl_gamma(prior.a, prior.b, a_q, (b_q)[s]) for s in 1:D])
        elbo = log_z - kl_ρ 
		el_s[i] = elbo

        if abs(elbo - elbo_prev) < tol
			println("Stopped at iteration: $i")
			el_s = el_s[1:i]
            break
		end

        elbo_prev = elbo

        if (i == max_iter)
			println("Warning: VB have not necessarily converged at $max_iter iterations with tolerance $tol")
		end
	end

	return inv(E_R), el_s
end

function test_vb_static(iter)
    T = 500  # Number of time steps
    rho = 0.0  # System evolution parameter, static seasonal model: rho = 0
    r = 0.5  # Observation noise variance
    A = [-1.0 -1.0 -1.0; 1.0 0.0 0.0; 0.0 1.0 0.0]  # State transition matrix
    C = [1.0 0.0 0.0]  # Emission matrix
    Q = Diagonal([rho, 0.0, 0.0])  # System evolution noise covariance
    R = [r]

    K = size(A, 1)
    x_1 = [0.50, 0.35, 0.15] # Satisfies identifiability constraint: sum to 0
	Random.seed!(123)
    y, x = gen_sea(A, C, R, Q, T, x_1)

    prior = HPP_D(0.01, 0.01, 0.01, 0.01, zeros(K), Matrix{Float64}(I, K, K))
    @time E_R, els = vb_ss(y, A, C, prior, iter)

    p = plot(els, label="elbo")
    display(p)
    μs, _ = vb_e_static(y, A, C, inv(E_R), prior, true)

    println("R: ", E_R)
    println("MSE, MAD of seasonal components:", error_metrics(x, μs))
    plot_latent(x', μs')
end


function vi_elbo_comp(gen_fun = "S", max_T = 500, n=10)
    elbo_lg = zeros(n)
    elbo_ll = zeros(n)
    elbo_ss = zeros(n)

    for i in 1:n
        y, x_true = missing, missing
        Random.seed!(i)

        if gen_fun == "S"
            rho = 0.0  # System evolution parameter, static seasonal model: rho = 0
            r = 0.5  # Observation noise variance
            A = [-1.0 -1.0 -1.0; 1.0 0.0 0.0; 0.0 1.0 0.0]  # State transition matrix
            C = [1.0 0.0 0.0]  # Emission matrix
            Q = Diagonal([rho, 0.0, 0.0])  # System evolution noise covariance
            R = [r]
            x_1 = [0.50, 0.35, 0.15] # Satisfies identifiability constraint: sum to 0
           
            y, x_true = gen_sea(A, C, R, Q, max_T, x_1)
            K = size(A, 1)
            prior = HPP_D(0.01, 0.01, 0.01, 0.01, zeros(K), Matrix{Float64}(I, K, K))

            _, el_ss = vb_ss(y, A, C, prior)
            elbo_ss[i] = el_ss[end]

            hpp_ll = Priors_ll(0.1, 0.1, 0.1, 0.1, 0.0, 1.0)
            _, _, el_ll, _ = vb_ll_c(vec(y), hpp_ll)
            elbo_ll[i] = el_ll[end]

            A_lg = [1.0 1.0; 0.0 1.0]
            C_lg = [1.0 0.0]
            K = size(A_lg, 1)
            prior_lg = HPP_D(0.1, 0.1, 0.1, 0.1, zeros(K), Matrix{Float64}(I, K, K))
            _, _, el_lg, _ = vbem_lg_c(y, A_lg, C_lg, prior_lg)
            elbo_lg[i] = el_lg[end]
        end

        if gen_fun == "LL"
            y, x_true = LocalLevel.gen_data(1.0, 1.0, 1.0, 1.0, 0.0, 1.0, max_T)
            hpp_ll = Priors_ll(0.1, 0.1, 0.1, 0.1, 0.0, 1.0)
            _, _, el_ll, _ = vb_ll_c(y, hpp_ll)
            elbo_ll[i] = el_ll[end]

            A_lg = [1.0 1.0; 0.0 1.0]
            C_lg = [1.0 0.0]
            K = size(A_lg, 1)
            prior_lg = HPP_D(0.1, 0.1, 0.1, 0.1, zeros(K), Matrix{Float64}(I, K, K))
            _, _, el_lg, _ = vbem_lg_c(reshape(y, 1, :), A_lg, C_lg, prior_lg)
            elbo_lg[i] = el_lg[end]

            A = [-1.0 -1.0 -1.0; 1.0 0.0 0.0; 0.0 1.0 0.0]  # State transition matrix
            C = [1.0 0.0 0.0]  # Emission matrix
            K = size(A, 1)
            prior = HPP_D(0.01, 0.01, 0.01, 0.01, zeros(K), Matrix{Float64}(I, K, K))
            _, el_ss = vb_ss(reshape(y, 1, :), A, C, prior)
            elbo_ss[i] = el_ss[end]
        end

        if gen_fun == "LT"
            A_lg = [1.0 1.0; 0.0 1.0]
            C_lg = [1.0 0.0]
            Q = Diagonal([1.0, 1.0])
            R = [0.1]
            K = size(A_lg, 1)
            μ_0 = zeros(K)
            Σ_0 = Diagonal(ones(K))
            y, x_true = LinearGrowth.gen_data(A_lg, C_lg, Q, R, μ_0, Σ_0, max_T)
            prior_lg = HPP_D(0.1, 0.1, 0.1, 0.1, zeros(K), Matrix{Float64}(I, K, K))
            _, _, el_lg, _ = vbem_lg_c(y, A_lg, C_lg, prior_lg)
            elbo_lg[i] = el_lg[end]

            hpp_ll = Priors_ll(0.1, 0.1, 0.1, 0.1, 0.0, 1.0)
            _, _, el_ll, _ = vb_ll_c(vec(y), hpp_ll)
            elbo_ll[i] = el_ll[end]

            A = [-1.0 -1.0 -1.0; 1.0 0.0 0.0; 0.0 1.0 0.0]  # State transition matrix
            C = [1.0 0.0 0.0]  # Emission matrix
            K = size(A, 1)
            prior = HPP_D(0.01, 0.01, 0.01, 0.01, zeros(K), Matrix{Float64}(I, K, K))
            _, el_ss = vb_ss(reshape(y, 1, :), A, C, prior)
            elbo_ss[i] = el_ss[end]
        end

    end

    # println(elbo_ss)
    # println(elbo_ll)
    # println(elbo_lg)

    println("--- FINAL ElBO mean: ---")
	println("\tLL:", mean(elbo_ll))
	println("\tLT:", mean(elbo_lg))
	println("\tSS:", mean(elbo_ss))
    
    groups = repeat(["LLM", "LTM", "SM"], inner = length(elbo_ll))
    all_elbos = vcat(elbo_ll, elbo_lg, elbo_ss)

    p = dotplot(groups, all_elbos, group=groups, color=[:blue :orange :green], label="", ylabel="ELBO", legend=false)
    title!(p, "ELBO Model Selection, data:$gen_fun")
    display(p)
end

vi_elbo_comp("S")

#vi_elbo_comp("LL")

#vi_elbo_comp("LT")
