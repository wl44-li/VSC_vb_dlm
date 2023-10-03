include("kl_optim.jl")

begin
	using Distributions, Plots, Random
	using LinearAlgebra
	using StatsFuns
	using MCMCChains
	using Statistics
	using DataFrames
	using StatsPlots
	using SpecialFunctions
	using StatsBase
end

function vb_m_step(y, hss::HSS, hpp::HPP_D, A::Array{Float64, 2}, C::Array{Float64, 2})
    D, T = size(y)
    K = size(A, 1)
	
	G = y*y' - 2 * C * hss.S_C + C * hss.W_C * C'
    a_ = hpp.a + 0.5 * T
	a_s = a_ * ones(D)
    b_s = [hpp.b + 0.5 * G[i, i] for i in 1:D]
	q_ρ = Gamma.(a_s, 1 ./ b_s)
	Exp_R⁻¹ = diagm(mean.(q_ρ))
	
	α_ = hpp.α + 0.5 * T
	α_s = α_ * ones(K)
	H_22 = hss.W_C[2, 2] + hss.W_A[2, 2] - 2*hss.S_A[2, 2]
	H_11 = hss.W_C[1, 1] - 2*hss.S_A[1, 1] + hss.W_A[1, 1] - 2*hss.S_A[2, 1] + 2*hss.W_A[1, 2] + hss.W_A[2, 2]

	H = Diagonal([H_11, H_22])
	β_s = [hpp.β + 0.5 * H[i, i] for i in 1:K]
	q_𝛐 = Gamma.(α_s, 1 ./ β_s)	
	Exp_Q⁻¹= diagm(mean.(q_𝛐))
	return Exp_R⁻¹, Exp_Q⁻¹, Q_Gamma(a_, b_s, α_, β_s)
end

function forward_(y::Array{Float64, 2}, A::Array{Float64, 2}, C::Array{Float64, 2}, E_R, E_Q::Array{Float64, 2}, prior::HPP_D)
    P, T = size(y)
    K = size(A, 1)
    μ_0, Σ_0 = prior.μ_0, prior.Σ_0
    
    # Initialize the filtered means and covariances
    μ_f = zeros(K, T)
    Σ_f = zeros(K, K, T)
    f_s = zeros(P, T)
	S_s = zeros(P, P, T)
    # Set the initial filtered mean and covariance to their prior values
	A_1 = A * μ_0
	R_1 = A * Σ_0 * A' + E_Q
	
	f_1 = C * A_1
	S_1 = C * R_1 * C' + E_R
	f_s[:, 1] = f_1
	S_s[:, :, 1] = S_1
	
    μ_f[:, 1] = A_1 + R_1 * C'* inv(S_1) * (y[:, 1] - f_1)
    Σ_f[:, :, 1] = R_1 - R_1*C'*inv(S_1)*C*R_1 
    
    # Forward pass (kalman filter)
    for t = 2:T
        μ_p = A * μ_f[:, t-1]
        Σ_p = A * Σ_f[:, :, t-1] * A' + E_Q
		f_t = C * μ_p
		S_t = C * Σ_p * C' + E_R
		f_s[:, t] = f_t
		S_s[:, :, t] = S_t
		μ_f[:, t] = μ_p + Σ_p * C' * inv(S_t) * (y[:, t] - f_t)
		Σ_f[:, :, t] = Σ_p - Σ_p * C' * inv(S_t) * C * Σ_p
    end
	
    log_z = sum(logpdf(MvNormal(f_s[:, i], Symmetric(S_s[:, :, i])), y[:, i]) for i in 1:T)
    return μ_f, Σ_f, log_z
end

function backward_(μ_f::Array{Float64, 2}, Σ_f::Array{Float64, 3}, A::Array{Float64, 2}, E_Q::Array{Float64, 2}, prior)
    K, T = size(μ_f)
    
    μ_s = zeros(K, T)
    Σ_s = zeros(K, K, T)
    Σ_s_cross = zeros(K, K, T)
    
    μ_s[:, T] = μ_f[:, T]
    Σ_s[:, :, T] = Σ_f[:, :, T]
    
    # Backward pass
    for t = T-1:-1:1
        J_t = Σ_f[:, :, t] * A' / (A * Σ_f[:, :, t] * A' + E_Q)

        μ_s[:, t] = μ_f[:, t] + J_t * (μ_s[:, t+1] - A * μ_f[:, t])
        Σ_s[:, :, t] = Σ_f[:, :, t] + J_t * (Σ_s[:, :, t+1] - A * Σ_f[:, :, t] * A' - E_Q) * J_t'

		Σ_s_cross[:, :, t+1] = J_t * Σ_s[:, :, t+1]
    end
	Σ_s_cross[:, :, 1] = inv(I + A'*A) * A' * Σ_s[:, :, 1]

	J_0 = prior.Σ_0 * A' / (A * prior.Σ_0 * A' + E_Q)
	μ_s0 = prior.μ_0 + J_0 * (μ_s[:, 1] -  A * prior.μ_0)
	Σ_s0 = prior.Σ_0 + J_0 * (Σ_s[:, :, 1] - A * prior.Σ_0 * A' - E_Q) * J_0'
	
    return μ_s, Σ_s, μ_s0, Σ_s0, Σ_s_cross
end

function vb_e_step(y::Array{Float64, 2}, A::Array{Float64, 2}, C::Array{Float64, 2}, E_R, E_Q::Array{Float64, 2}, prior::HPP_D)
    # Run the forward pass
    μ_f, Σ_f, log_Z = forward_(y, A, C, E_R, E_Q, prior)

    # Run the backward pass
    μ_s, Σ_s, μ_s0, Σ_s0, Σ_s_cross = backward_(μ_f, Σ_f, A, E_Q, prior)

    # Compute the hidden state sufficient statistics
    W_C = sum(Σ_s, dims=3)[:, :, 1] + μ_s * μ_s'
    W_A = sum(Σ_s[:, :, 1:end-1], dims=3)[:, :, 1] + μ_s[:, 1:end-1] * μ_s[:, 1:end-1]'
    S_C = μ_s * y'
    S_A = sum(Σ_s_cross, dims=3)[:, :, 1] + μ_s[:, 1:end-1] * μ_s[:, 2:end]'
    W_Y = y * y'

	# Return the hidden state sufficient statistics
    return HSS(W_C, W_A, S_C, S_A), μ_s0, Σ_s0, log_Z
end

function vbem_lg(y::Array{Float64, 2}, A::Array{Float64, 2}, C::Array{Float64, 2}, prior::HPP_D, max_iter=100)
	
	hss = HSS(ones(size(A)), ones(size(A)), ones(size(C')), ones(size(A)))
	E_R_inv, E_Q_inv = missing, missing

	for i in 1:max_iter
		E_R_inv, E_Q_inv, _ = vb_m_step(y, hss, prior, A, C)
				
		hss, _ = vb_e_step(y, A, C, inv(E_R_inv), inv(E_Q_inv), prior)
	end
	
	return E_R_inv, E_Q_inv
end

function vbem_his_plot(y::Array{Float64, 2}, A::Array{Float64, 2}, C::Array{Float64, 2}, prior::HPP_D, max_iter=100)
    P, T = size(y)
    K, _ = size(A)

    W_C = zeros(K, K)
    W_A = zeros(K, K)
    S_C = zeros(K, P)
    S_A = zeros(K, K)
    hss = HSS(W_C, W_A, S_C, S_A)
	
    # Initialize the history of E_R and E_Q
    E_R_history = zeros(P, max_iter)
    E_Q_history = zeros(K, K, max_iter)

    # Repeat until convergence
    for iter in 1:max_iter
		E_R_inv, E_Q_inv = vb_m_step(y, hss, prior, A, C)
				
		hss, _ = vb_e_step(y, A, C, inv(E_R_inv), inv(E_Q_inv), prior)

        # Store the history of E_R and E_Q
        E_R_history[:, iter] = E_R_inv
        E_Q_history[:, :, iter] = E_Q_inv
    end

	p1 = plot(title = "Learning of R")
    for i in 1:P
        plot!(5:max_iter, [E_R_history[i, t] for t in 5:max_iter], label = "E_R[$i]")
    end

    p2 = plot(title = "Learning of Q")
    for i in 1:K
        plot!(5:max_iter, [E_Q_history[i, i, t] for t in 5:max_iter], label = "E_Q[$i, $i]")
    end
	
	plot(p1, p2, layout = (1, 2))
end

function vbem_lg_c(y, A::Array{Float64, 2}, C::Array{Float64, 2}, prior, hp_learn=false, max_iter=200, tol=5e-3)

	D, _ = size(y)
	K = size(A, 1)
	hss = HSS(ones(size(A)), ones(size(A)), ones(size(C')), ones(size(A)))
	E_R_inv, E_Q_inv = missing, missing
	elbo_prev = -Inf
	el_s = zeros(max_iter)
	
	for i in 1:max_iter
		E_R_inv, E_Q_inv, Q_gam = vb_m_step(y, hss, prior, A, C)
		hss, μ_s0, Σ_s0, log_Z = vb_e_step(y, A, C, inv(E_R_inv), inv(E_Q_inv), prior)
		
		kl_ρ = sum([kl_gamma(prior.a, prior.b, Q_gam.a, (Q_gam.b)[s]) for s in 1:D])
		kl_𝛐 = sum([kl_gamma(prior.α, prior.β, Q_gam.α, (Q_gam.β)[s]) for s in 1:K])
		
		elbo = log_Z - kl_ρ - kl_𝛐
		el_s[i] = elbo
		
		if (hp_learn)
			if (i%5 == 0) 
				a_, b_, α_, β_ = update_hyp_D(prior, Q_gam)
				prior = HPP_D(α_, β_, a_, b_, μ_s0, Σ_s0)
			end
		end
		
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
	
	return inv(E_R_inv), inv(E_Q_inv), el_s
end

function main()
	A_lg = [1.0 1.0; 0.0 1.0]
    C_lg = [1.0 0.0]
	Q_lg = Diagonal([0.05, 0.03])
	R_lg = [0.1]
	μ_0 = [0.0, 0.0]
	Σ_0 = Diagonal([1.0, 1.0])
	Random.seed!(111)
	T = 500
	y, x_true = gen_data(A_lg, C_lg, Q_lg, R_lg, μ_0, Σ_0, T)
	K = size(A_lg, 1)
	prior = HPP_D(0.01, 0.01, 0.01, 0.01, zeros(K), Matrix{Float64}(I, K, K))

	for t in [false, true]
		println("\nHyperparam optimisation: $t")
		@time R, Q, elbos = vbem_lg_c(y, A_lg, C_lg, prior, t)
		display(plot(elbos, label = "elbo", title = "ElBO Progression Linear Growth"))

		μs_f, σs_f2 = forward_(y, A_lg, C_lg, R, Q, prior)
		μs_s, _, _ = backward_(μs_f, σs_f2, A_lg, Q, prior)
		println("MSE, MAD of VB X: ", error_metrics(x_true, μs_s))
	end
end

main()

# PLUTO_PROJECT_TOML_CONTENTS = """
# [deps]
# DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
# Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
# LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
# MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
# PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
# Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
# PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
# Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
# SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
# Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
# StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
# StatsFuns = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
# StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

# [compat]
# DataFrames = "~1.5.0"
# Distributions = "~0.25.96"
# MCMCChains = "~6.0.3"
# PDMats = "~0.11.17"
# Plots = "~1.38.16"
# PlutoUI = "~0.7.51"
# SpecialFunctions = "~2.2.0"
# StatsBase = "~0.34.0"
# StatsFuns = "~1.3.0"
# StatsPlots = "~0.15.5"
# """
