include("kl_optim.jl")

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
    return Exp_R⁻¹
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
	
	f_1 = C * A_1
	S_1 = C * R_1 * C' + E_R
	f_s[:, 1] = f_1
	S_s[:, :, 1] = S_1
	
    μ_f[:, 1] = A_1 + R_1 * C'* inv(S_1) * (y[:, 1] - f_1)
    Σ_f[:, :, 1] = R_1 - R_1*C'*inv(S_1)*C*R_1 
    
    # Forward pass (kalman filter)
    for t = 2:T
        # Predicted state mean and covariance
        μ_p = A * μ_f[:, t-1]
        Σ_p = A * Σ_f[:, :, t-1] * A'

		# marginal y - normalization
		f_t = C * μ_p
		S_t = C * Σ_p * C' + E_R
		f_s[:, t] = f_t
		S_s[:, :, t] = S_t
		
		# Filtered state mean and covariance (2.8a - 2.8c DLM with R)
		μ_f[:, t] = μ_p + Σ_p * C' * inv(S_t) * (y[:, t] - f_t)
		Σ_f[:, :, t] = Σ_p - Σ_p * C' * inv(S_t) * C * Σ_p	
    end

    if (smooth_out)
		return μ_f, Σ_f
	end

    W_C = sum(Σ_f[:, :, t] + μ_f[:, t] * μ_f[:, t]' for t in 1:T)
	S_C = sum(μ_f[:, t] * y[:, t]' for t in 1:T)

    return W_C, S_C
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

function vb_ss(ys, A, C, prior::HPP_D, max_iter = 200)

    E_R = missing

    W_C, S_C = ones(size(A)), ones(size(C'))
    for _ in 1:max_iter
		E_R = vb_m_static(ys, S_C, W_C, prior, C)
				
		W_C, S_C = vb_e_static(ys, A, C, inv(E_R), prior)
	end

	return inv(E_R)
end

function test_vb_static(iter)
    T = 500  # Number of time steps
    rho = 0.0  # System evolution parameter, static seasonal model: rho = 0
    r = 0.1  # Observation noise variance
    A = [-1.0 -1.0 -1.0; 1.0 0.0 0.0; 0.0 1.0 0.0]  # State transition matrix
    C = [1.0 0.0 0.0]  # Emission matrix
    Q = Diagonal([rho, 0.0, 0.0])  # System evolution noise covariance
    R = [r]

    K = size(A, 1)
    x_1 = [0.50, 0.35, 0.15] # Satisfies identifiability constraint: sum to 0
	Random.seed!(123)
    y, x = gen_sea(A, C, R, Q, T, x_1)

    prior = HPP_D(0.01, 0.01, 0.01, 0.01, zeros(K), Matrix{Float64}(I, K, K))
    @time E_R = vb_ss(y, A, C, prior, iter)

    μs, _ = vb_e_static(y, A, C, inv(E_R), prior, true)

    println("R: ", E_R)
    println("mse, mad of seasonal components:", error_metrics(x, μs))
    plot_latent(x', μs')
end

test_vb_static(50)

