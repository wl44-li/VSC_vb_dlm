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
	using PDMats
	using StatsBase
	using Dates
	using DataFrames
end

function gen_data(A, C, Q, R, m_0, C_0, T)
	K, _ = size(A)
	D, _ = size(C)
	x = zeros(K, T+1)
	y = zeros(D, T)

	x[:, 1] = zeros(K)
	x[:, 2] = rand(MvNormal(A*m_0, A'*C_0*A + Q))
	y[:, 1] = C * x[:, 2] + rand(MvNormal(zeros(D), R))

	for t in 2:T
		x[:, t+1] = A * x[:, t] + rand(MvNormal(zeros(K), Q))
		y[:, t] = C * x[:, t+1] + rand(MvNormal(zeros(D), R)) 
	end
	return y, x
end

"""
MCMC
"""
function forward_filter(Ys, A, C, R, Q, m_0, C_0)
	"""
	A : State transition (K X K)
	C : Emission (P X K)
	R : Observation noise (P X P)
	Q : System noise (diagonal) (K X K)
	"""
	_, T = size(Ys)
	K, _ = size(A)
	
	# Initialize, using "DLM with R" notation
	ms = zeros(K, T+1)
	Cs = zeros(K, K, T+1)

	ms[:, 1] = m_0
	Cs[:, :, 1] = C_0
	
	# one-step ahead latent distribution, used in backward sampling
	a_s = zeros(K, T)
	Rs = zeros(K, K, T)

	for t in 1:T
		# Prediction
		a_s[:, t] = a_t = A * ms[:, t]
		Rs[:, :, t] = R_t = A * Cs[:, :, t] * A' + Q #W
		
		# Update
		f_t = C * a_t
		Q_t = C * R_t * C' + R #V

		# Kalman Filter 
		ms[:, t+1] = a_t + R_t * C' * inv(Q_t) * (Ys[:, t] - f_t)
		Cs[:, :, t+1] = R_t - R_t * C' * inv(Q_t) * C * R_t
	end
	return ms, Cs, a_s, Rs
end

function ffbs_x(Ys, A, C, R, Q, m_0, C_0)
	K, _ = size(A)
	_, T = size(Ys)

	ms, Cs, a_s, Rs = forward_filter(Ys, A, C, R, Q, m_0, C_0)
	X = zeros(K, T+1)

	# temp hack
	lambda = 5e-6
	X[:, end] = rand(MvNormal(ms[:, end], Symmetric(Cs[:, :, end] + lambda * I)))

	# Backward Sampling (h_t, H_t)
	for t in T:-1:1
		C_t = Cs[:, :, t]
		h_t = ms[:, t] + C_t * A' * inv(Rs[:, :, t])*(X[:, t+1] - a_s[:, t])
		H_t = C_t - C_t * A' * inv(Rs[:, :, t]) * A * C_t
		X[:, t] = rand(MvNormal(h_t, Symmetric(H_t + lambda * I)))
	end

	return X
end

# Multi-variate DLM with full R
function sample_R_(y, x, C, v_0, S_0)
    T = size(y, 2)
	x = x[:, 2:end]
    residuals = [y[:, t] - C * x[:, t] for t in 1:T]
	SS_y = sum([residuals[t] * residuals[t]' for t in 1:T])
	
    scale_posterior = S_0 + SS_y .* 0.5
    v_p = v_0 + 0.5 * T

	S_p = PDMat(Symmetric(inv(scale_posterior)))
	R⁻¹ = rand(Wishart(v_p, S_p))
    return inv(R⁻¹)
end

function test_full_R(sd, T = 100)
	A = [1.0 0.0; 0.0 1.0]
	C = [1.0 0.0; 0.0 1.0]
	Q = Diagonal([1.0, 1.0])
	R = [5.0 1.0; 1.0 4.0] # Full-cov R

	println("R True:")
	show(stdout, "text/plain", R)
	println()
	m_0 = [0.0, 0.0]
	C_0 = Diagonal([1.0, 1.0])
	Random.seed!(sd)
	y, x_true = gen_data(A, C, Q, R, m_0, C_0, T)

	P, T = size(y)
	v_0 = P + 1.0
	S_0 = Matrix{Float64}(I * 0.001, P, P)
	R = sample_R_(y, x_true, C, v_0, S_0)
	println("\nR Sample", R)
end

# test_full_R(123, 100)
# println()
# test_full_R(123, 2000)

function gibbs_dlm_cov(y, A, C, mcmc=10000, burn_in=5000, thinning=1, debug=false)
	P, T = size(y)
	K = size(A, 2)
	
	m_0, C_0 = zeros(K), Matrix{Float64}(I * 1e7, K, K)
	
	v_0 = P + 1.0 
	S_0 = Matrix{Float64}(1e-3 * I, P, P)
	α, β = 2, 1e-3

	"""
	Initilise both R, Q to I (DLM with R setting)
	"""
	R = Diagonal(ones(P))
    Q = Diagonal(ones(K))
	
	n_samples = Int.(mcmc/thinning)
	samples_X = zeros(n_samples, K, T)
	samples_Q = zeros(n_samples, K, K)
	samples_R = zeros(n_samples, P, P)
	
	for i in 1:mcmc+burn_in
		x = ffbs_x(y, A, C, R, Q, m_0, C_0)
		Q = sample_Q(x, A, α, β)
		R = sample_R_(y, x, C, v_0, S_0)
		x = x[:, 2:end]

		if i > burn_in && mod(i - burn_in, thinning) == 0
			index = div(i - burn_in, thinning)
		    samples_X[index, :, :] = x
			samples_Q[index, :, :] = Q
		    samples_R[index, :, :] = R
		end
	end

	return samples_X, samples_Q, samples_R
end

function test_gibbs_cov(y, x_true=nothing, mcmc=10000, burn_in=5000, thin=1)
	A = [1.0 0.0; 0.0 1.0]
	C = [1.0 0.0; 0.0 1.0]
	D, _ = size(y)
	K = size(A, 1)

	println("\n--- MCMC : R Full Covariances ---")
	n_samples = Int.(mcmc/thin)
	println("\n--- n_samples: $n_samples, burn-in: $burn_in, thinning: $thin ---")

	@time Xs_samples, Qs_samples, Rs_samples = gibbs_dlm_cov(y, A, C, mcmc, burn_in, thin)
	Q_chain = Chains(reshape(Qs_samples, n_samples, K^2))
	R_chain = Chains(reshape(Rs_samples, n_samples, D^2))

	summary_stats_q = summarystats(Q_chain)
	summary_stats_r = summarystats(R_chain)
	summary_df_q = DataFrame(summary_stats_q)
	summary_df_r = DataFrame(summary_stats_r)
	summary_df_q = summary_df_q[[1,4], :]
	println("Q summary stats: ", summary_df_q)
	println()
	println("R summary stats: ", summary_df_r)
	println()

	xs_m = mean(Xs_samples, dims=1)[1, :, :]
	if x_true !== nothing
		println("MSE, MAD of MCMC X mean: ", error_metrics(x_true[:, 2:end], xs_m))
	end
end

"""
VBEM
"""

begin
	struct Prior
	    ν_R::Float64
	    W_R::Array{Float64, 2}
	    a::Float64
	   	b::Float64
	    m_0::Array{Float64, 1}
	    C_0::Array{Float64, 2}
	end

	struct Q_Wishart
		ν_R_q
		W_R_q
		a_q
		b_q
	end
end

function vb_m_step(y::Array{Float64, 2}, hss::HSS, prior::Prior, A::Array{Float64, 2}, C::Array{Float64, 2})
    _, T = size(y)
    K = size(A, 2) 

	# Wishart posterior of R
    ν_Rn = prior.ν_R + T
	W_Rn_inv = inv(prior.W_R) + y*y' - hss.S_C * C' - C * hss.S_C' + C * hss.W_C * C'
    
	# Gamma posterior of Q
    H = hss.W_C - hss.S_A * A' - A * hss.S_A' + A * hss.W_A * A'
	a_ = prior.a + 0.5 * T
	a_s = a_ * ones(K) # shape
    b_s = [prior.b + 0.5 * H[i, i] for i in 1:K] # rate
	q_𝛐 = Gamma.(a_s, 1 ./ b_s)	
	Exp_Q⁻¹= diagm(mean.(q_𝛐))

	# q_ = InverseGamma.(a_s, b_s)	
	# E_Q = diagm(mean.(q_))

	# return E_R, E_Q
	return W_Rn_inv ./ ν_Rn, inv(Exp_Q⁻¹), Q_Wishart(ν_Rn, inv(W_Rn_inv), a_, b_s)
end

function forward_(y::Array{Float64, 2}, A::Array{Float64, 2}, C::Array{Float64, 2}, E_R::Array{Float64, 2}, E_Q::Array{Float64, 2}, m_0, C_0)
    _, T = size(y)
    K = size(A, 1)
        
    μ_f = zeros(K, T+1)
    Σ_f = zeros(K, K, T+1)
    A_s = zeros(K, T)
	R_s = zeros(K, K, T)
	μ_f[:, 1], Σ_f[:, :, 1] = m_0, C_0

	log_Z = 0.0
    for t in 1:T # Forward pass (Kalman Filter)
        A_s[:, t] = A_t = A * μ_f[:, t]
        R_s[:, :, t] = R_t = A * Σ_f[:, :, t] * A' + E_Q

		# marginal y - normalization
		f_t = C * A_t
		Q_t = C * R_t * C' + E_R
		log_Z += logpdf(MvNormal(f_t, Symmetric(Q_t)), y[:, t])

		μ_f[:, t+1] = A_t + R_t * C' * inv(Q_t) * (y[:, t] - f_t)
		Σ_f[:, :, t+1] = R_t - R_t * C' * inv(Q_t) * C * R_t
    end
	
    return μ_f, Σ_f, A_s, R_s, log_Z
end

function backward_(A::Array{Float64, 2}, μ_f::Array{Float64, 2}, Σ_f::Array{Float64, 3}, A_s, R_s)
    K, T = size(A_s)
    μ_s = zeros(K, T+1)
    Σ_s = zeros(K, K, T+1)
    Σ_s_cross = zeros(K, K, T)
    
    # Set the final (t=T) smoothed mean and co-variance to filtered values
    μ_s[:, end], Σ_s[:, :, end] = μ_f[:, end], Σ_f[:, :, end]
    
    for t in T:-1:1  # Backward pass, Kalman Smoother (s_t, S_t)
		J_t = Σ_f[:, :, t] * A' * inv(R_s[:, :, t])
		μ_s[:, t] = μ_f[:, t] + J_t * (μ_s[:, t+1] - A_s[:, t])
		Σ_s[:, :, t] = Σ_f[:, :, t] - J_t * (R_s[:, :, t] - Σ_s[:, :, t+1]) * inv(R_s[:, :, t]) * A * Σ_f[:, :, t]
		Σ_s_cross[:, :, t] = J_t * Σ_s[:, :, t+1]
    end
	
    return μ_s, Σ_s, Σ_s_cross
end

function vb_e_step(y::Array{Float64, 2}, A::Array{Float64, 2}, C::Array{Float64, 2}, E_R::Array{Float64, 2}, E_Q::Array{Float64, 2}, prior::Prior)
	# Forward-Backward
	μ_f, Σ_f, A_s, R_s, log_Z = forward_(y, A, C, E_R, E_Q, prior.m_0, prior.C_0)
    μ_s, Σ_s, Σ_s_cross = backward_(A, μ_f, Σ_f, A_s, R_s)

    # Compute the hidden state sufficient statistics [ Beale 5.3, page 183 ]
    W_C = sum(Σ_s[:, :, 2:end], dims=3)[:, :, 1] + μ_s[:, 2:end] * μ_s[:, 2:end]'
    W_A = sum(Σ_s[:, :, 1:end-1], dims=3)[:, :, 1] + μ_s[:, 1:end-1] * μ_s[:, 1:end-1]'
	S_C = μ_s[:, 2:end] * y'
    S_A = sum(Σ_s_cross, dims=3)[:, :, 1] + μ_s[:, 1:end-1] * μ_s[:, 2:end]'

	# Return hidden state sufficient statistics
    return HSS(W_C, W_A, S_C, S_A), log_Z
end

function vbem_c(y::Array{Float64, 2}, A::Array{Float64, 2}, C::Array{Float64, 2}, prior::Prior, max_iter=1000, tol=1e-4; init="gibbs_full", debug=false)
	K = size(A, 2)
	E_R, E_Q = missing, missing
	elbo_prev = -Inf
	el_s = zeros(max_iter)
	hss = missing

	if init == "fixed"
		hss = HSS(ones(size(A)), ones(size(A)), ones(size(C)), ones(size(A)))
	end

	if init == "gibbs"
		println("--- Using Gibbs 1-step initilaization ---")
		prior_d = HPP_D(2, 1e-3, 2, 1e-3, zeros(K), Matrix{Float64}(I * 1e7, K, K))
		_, S_Q, S_R = gibbs_diag(y, A, C, prior_d, 1, 0, 1)
		Q_init, R_init = S_Q[1, :, :], S_R[1, :, :]
		hss, _ = vb_e_step(y, A, C, R_init, Q_init, prior)
		if debug
			println("Q_init: ", Q_init)
			println("R_init: ", R_init)
			println("W_C, W_A, S_C, S_A :", hss)
		end
	end

	if init == "gibbs_full"
		println("--- Using Gibbs (Full R) 1-step initilaization ---")
		_, S_Q, S_R = gibbs_dlm_cov(y, A, C, 1, 0, 1)
		Q_init, R_init = S_Q[1, :, :], S_R[1, :, :]
		hss, _ = vb_e_step(y, A, C, R_init, Q_init, prior)
		if debug
			println("Q_init: ", Q_init)
			println("R_init: ", R_init)
			println("W_C, W_A, S_C, S_A :", hss)
		end
	end

	for i in 1:max_iter
		E_R, E_Q, Q_Wi = vb_m_step(y, hss, prior, A, C)
		hss, log_Z = vb_e_step(y, A, C, E_R, E_Q, prior)

		kl_Wi = kl_Wishart(Q_Wi.ν_R_q, Q_Wi.W_R_q, prior.ν_R, prior.W_R)
		kl_gam = sum([kl_gamma(prior.a, prior.b, Q_Wi.a_q, (Q_Wi.b_q)[s]) for s in 1:K])
		elbo = log_Z - kl_Wi - kl_gam
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

	return E_R, E_Q, el_s
end

# Restrict R, Q as diagonal matrices
function vb_m_diag(y, hss::HSS, hpp::HPP_D, A::Array{Float64, 2}, C::Array{Float64, 2})
    D, T = size(y)
    K = size(A, 1)

	G = y*y' - hss.S_C * C' - C * hss.S_C' + C * hss.W_C * C'
    a_ = hpp.a + 0.5 * T
	a_s = a_ * ones(D) # shape
    b_s = [hpp.b + 0.5 * G[i, i] for i in 1:D] # rate
	q_ρ = Gamma.(a_s, 1 ./ b_s) # Julia Gamma: param (shape, scale) (1/rate)
	Exp_R⁻¹ = diagm(mean.(q_ρ))
	
    H = hss.W_C - hss.S_A * A' - A * hss.S_A' + A * hss.W_A * A'
	α_ = hpp.α + 0.5 * T
	α_s = α_ * ones(K)
    β_s = [hpp.β + 0.5 * H[i, i] for i in 1:K]
	q_𝛐 = Gamma.(α_s, 1 ./ β_s)	
	Exp_Q⁻¹= diagm(mean.(q_𝛐))
	
	return Exp_R⁻¹, Exp_Q⁻¹, Q_Gamma(a_, b_s, α_, β_s)
end

function vb_e_diag(y, A::Array{Float64, 2}, C::Array{Float64, 2}, E_R, E_Q, prior::HPP_D)
    μ_f, Σ_f, A_s, R_s, log_Z = forward_(y, A, C, E_R, E_Q, prior.μ_0, prior.Σ_0)
    μ_s, Σ_s, Σ_s_cross = backward_(A, μ_f, Σ_f, A_s, R_s)

    W_C = sum(Σ_s[:, :, 2:end], dims=3)[:, :, 1] + μ_s[:, 2:end] * μ_s[:, 2:end]'
    W_A = sum(Σ_s[:, :, 1:end-1], dims=3)[:, :, 1] + μ_s[:, 1:end-1] * μ_s[:, 1:end-1]'
	
    S_C = μ_s[:, 2:end] * y'
    S_A = sum(Σ_s_cross, dims=3)[:, :, 1] + μ_s[:, 1:end-1] * μ_s[:, 2:end]'

    return HSS(W_C, W_A, S_C, S_A), μ_s[:, :, 1], Σ_s[:, :, 1], log_Z
end

# VBEM with Convergence
function vbem_c_diag(y, A::Array{Float64, 2}, C::Array{Float64, 2}, prior::HPP_D, hp_learn=false, max_iter=500, tol=1e-4; init = "gibbs", debug = false)
	D, _ = size(y)
	K = size(A, 1)
	E_R_inv, E_Q_inv = missing, missing
	elbo_prev = -Inf
	el_s = zeros(max_iter)
	hss = missing
	
	if init == "fixed"
		hss = HSS(ones(size(A)), ones(size(A)), ones(size(C)), ones(size(A)))
	end

	if init == "gibbs"
		println("--- Using Gibbs 1-step initilaization ---")
		_, S_Q, S_R = gibbs_diag(y, A, C, prior, 1, 0, 1)
		Q_init, R_init = S_Q[1, :, :], S_R[1, :, :]
		hss, _ = vb_e_diag(y, A, C, R_init, Q_init, prior)
		if debug
			println("Q_init: ", Q_init)
			println("R_init: ", R_init)
			println("W_C, W_A, S_C, S_A :", hss)
		end
	end

	for i in 1:max_iter
		E_R_inv, E_Q_inv, Q_gam = vb_m_diag(y, hss, prior, A, C)
		hss, μ_s0, Σ_s0, log_Z = vb_e_diag(y, A, C, inv(E_R_inv), inv(E_Q_inv), prior)
		
		kl_ρ = sum([kl_gamma(prior.a, prior.b, Q_gam.a, (Q_gam.b)[s]) for s in 1:D])
		kl_𝛐 = sum([kl_gamma(prior.α, prior.β, Q_gam.α, (Q_gam.β)[s]) for s in 1:K])
		
		elbo = log_Z - kl_ρ - kl_𝛐
		el_s[i] = elbo

		if abs(elbo - elbo_prev) < tol
			println("Stopped at iteration: $i")
			el_s = el_s[1:i]
            break
		end
		
		if (hp_learn)
			if (i%5 == 0) 
				a_, b_, α_, β_ = update_hyp_D(prior, Q_gam)
				prior = HPP_D(α_, β_, a_, b_, μ_s0, Σ_s0)
			end
		end

        elbo_prev = elbo

		if (i == max_iter)
			println("Warning: VB have not necessarily converged at $max_iter iterations with tolerance $tol")
		end
	end
	
	return inv(E_R_inv), inv(E_Q_inv), el_s
end

"""
MCMC (Diagonal R, Q)
"""
function sample_R(Xs, Ys, C, a_ρ, b_ρ)
    P, T = size(Ys)
	Xs = Xs[:, 2:end]
    ρ_sampled = zeros(P)
    for i in 1:P
        Y = Ys[i, :]
        a_post = a_ρ + T / 2 # shape
        b_post = b_ρ + 0.5 * sum((Y' - C[i, :]' * Xs).^2) # rate
        ρ_sampled[i] = rand(Gamma(a_post, 1 / b_post))
    end
    return diagm(1 ./ ρ_sampled)
end

function sample_Q(Xs, A, α_q, β_q)
    K, T = size(Xs)
    q_sampled = zeros(K)
    for i in 1:K
        X_diff = Xs[i, 2:end] - (A * Xs[:, 1:end-1])[i, :]
        α_post = α_q + (T-1) / 2
        β_post = β_q + 0.5 * sum(X_diff.^2)
        q_sampled[i] = rand(Gamma(α_post, 1 / β_post)) # sample precison
    end
    return diagm(1 ./ q_sampled)
end

function test_Gibbs_RQ(sd, T=500)
	A = [1.0 0.0; 0.0 1.0]
	C = [1.0 0.0; 0.0 1.0]
	Q = Diagonal([10.0, 8.0])
	R = Diagonal([1.0, 1.0])

	println("R True:")
	show(stdout, "text/plain", R)
	println()
	println("Q True:")
	show(stdout, "text/plain", Q)
	println()

	m_0 = [0.0, 0.0]
	C_0 = Diagonal([1.0, 1.0])
	Random.seed!(sd)
	y, x_true = gen_data(A, C, Q, R, m_0, C_0, T)
	prior = Q_Gamma(2, 0.001, 2, 0.001)

	R = sample_R(x_true, y, C, prior.a, prior.b)
	Q = sample_Q(x_true, A, prior.α, prior.β)

	println("R Sample:")
	show(stdout, "text/plain", R)
	println()
	println("Q Sample:")
	show(stdout, "text/plain", Q)
end

# Diagonal R, Q verfied (linear growth simple extension)
# test_Gibbs_RQ(111, 100)
# println()
# test_Gibbs_RQ(111, 1000)

function gibbs_diag(y, A, C, prior::HPP_D, mcmc=3000, burn_in=1500, thinning=1, debug=false)
	P, T = size(y)
	K = size(A, 2)
	m_0, C_0 = prior.μ_0, prior.Σ_0
	a, b, α, β = prior.a, prior.b, prior.α, prior.β
	
    R = Diagonal(ones(P))
    Q = Diagonal(ones(K))

	n_samples = Int.(mcmc/thinning)
	samples_X = zeros(n_samples, K, T)
	samples_Q = zeros(n_samples, K, K)
	samples_R = zeros(n_samples, P, P)
	
	for i in 1:mcmc+burn_in
		if debug
			println("MCMC diagonal R, Q debug, iteration : $i")
		end
		x = ffbs_x(y, A, C, R, Q, m_0, C_0)
		Q = sample_Q(x, A, α, β)
		R = sample_R(x, y, C, a, b)
		x = x[:, 2:end]

		if i > burn_in && mod(i - burn_in, thinning) == 0
			index = div(i - burn_in, thinning)
		    samples_X[index, :, :] = x
			samples_Q[index, :, :] = Q
		    samples_R[index, :, :] = R
		end
	end

	return samples_X, samples_Q, samples_R
end

function test_gibbs_diag(y, x_true=nothing, mcmc=10000, burn_in=5000, thin=1)
	A = [1.0 0.0; 0.0 1.0] 
	C = [1.0 0.0; 0.0 1.0]
	K = size(A, 1)
	D, _ = size(y)

	# DEBUG, different choices of prior params of Gamma
	prior = HPP_D(2, 1e-3, 2, 1e-3, zeros(K), Matrix{Float64}(I * 1e7, K, K))
	n_samples = Int.(mcmc/thin)

	println("--- MCMC Diagonal Covariances ---")
	@time Xs_samples, Qs_samples, Rs_samples = gibbs_diag(y, A, C, prior, mcmc, burn_in, thin)
	println("\n--- n_samples: $n_samples, burn-in: $burn_in, thinning: $thin ---")
	Q_chain = Chains(reshape(Qs_samples, n_samples, K^2))
	R_chain = Chains(reshape(Rs_samples, n_samples, D^2))

	summary_stats_q = summarystats(Q_chain)
	summary_stats_r = summarystats(R_chain)
	summary_df_q = DataFrame(summary_stats_q)
	summary_df_r = DataFrame(summary_stats_r)

	summary_df_q = summary_df_q[[1, 4], :]
	summary_df_r = summary_df_r[[1, 4], :]
	println("Q summary stats: ", summary_df_q)
	println()
	println("R summary stats: ", summary_df_r)

	xs_m = mean(Xs_samples, dims=1)[1, :, :]
	if x_true !== nothing
		println("\nMSE, MAD of MCMC X mean: ", error_metrics(x_true[:, 2:end], xs_m))
	end
end

function test_vb(y, x_true=nothing, hyperoptim=false; show_plot=false)
	A = [1.0 0.0; 0.0 1.0]
	C = [1.0 0.0; 0.0 1.0]
	K = size(A, 1)
	prior = HPP_D(2, 1e-3, 2, 1e-3, zeros(K), Matrix{Float64}(I * 1e7, K, K))
	println("--- VB with Diagonal Covariances ---")
	println("Hyperparam optimisation: $hyperoptim")
	@time R, Q, elbos = vbem_c_diag(y, A, C, prior, hyperoptim)

	if show_plot
		p = plot(elbos, label = "elbo", title = "ElBO Progression, Hyperparam optim: $hyperoptim")
		display(p)
		#plot_file_name = "$(splitext(basename(@__FILE__))[1])_$(Dates.format(now(), "yyyymmdd_HHMMSS")).svg"
		#savefig(p, joinpath(expanduser("~/Downloads/_graphs"), plot_file_name))
	end

	println("VB q(R): ")
	show(stdout, "text/plain", R)
	println("\n\n VB q(Q): ")
	show(stdout, "text/plain", Q)
	μs_f, σs_f2, A_s, R_s = forward_(y, A, C, R, Q, prior.μ_0, prior.Σ_0)
	μs_s, _, _ = backward_(A, μs_f, σs_f2, A_s, R_s)

	if x_true !== nothing
		println("\n\nMSE, MAD of VB X: ", error_metrics(x_true[:, 2:end], μs_s[:, 2:end]))
	end

	# VBEM with full R co-variance
	D, _ = size(y)
	W_R = Matrix{Float64}(I * 1e3, D, D)
	prior = Prior(D + 1.0, W_R, 2, 1e-3, zeros(K), Matrix{Float64}(I * 1e7, K, K))
	println("\n--- VB with Full Co-variances R ---")
	@time R, Q, elbos = vbem_c(y, A, C, prior)
	println("VB q(R): ")
	show(stdout, "text/plain", R)
	println("\n\nVB q(Q): ")
	show(stdout, "text/plain", Q)

	if show_plot
		p = plot(elbos, label = "elbo", title = "ElBO progression, full R")
		display(p)
		#plot_file_name = "$(splitext(basename(@__FILE__))[1])_$(Dates.format(now(), "yyyymmdd_HHMMSS")).svg"
		#savefig(p, joinpath(expanduser("~/Downloads/_graphs"), plot_file_name))
	end

	μs_f, σs_f2, A_s, R_s = forward_(y, A, C, R, Q, prior.m_0, prior.C_0)
    μs_s, _, _ = backward_(A, μs_f, σs_f2, A_s, R_s)
	if x_true !== nothing
		println("\n\nMSE, MAD of VB X: ", error_metrics(x_true, μs_s))
	end
end

function test_data(rnd, max_T = 500)
	A = [1.0 0.0; 0.0 1.0]
	C = [1.0 0.0; 0.0 1.0]
	Q = Diagonal([10.0, 10.0]) # Diagonal Q
	R = [10.0 1.0; 1.0 10.0] # Full-cov R
	m_0 = [0.0, 0.0]
	C_0 = Diagonal([1.0, 1.0])
	Random.seed!(rnd)
	y, x_true = gen_data(A, C, Q, R, m_0, C_0, max_T)
	return y, x_true
end

function com_vb_gibbs()
	#seeds = [199, 188, 233, 234, 236]
	seeds = [111, 108, 134, 123, 105]
	for sd in seeds
		y, x_true = test_data(sd)
		println("--- Seed: $sd ---")
		test_gibbs_diag(y, x_true, 20000, 10000, 1)
		test_gibbs_cov(y, x_true, 20000, 10000, 1)
		test_vb(y, x_true, show_plot=true)
		println()
	end
end

function out_txt()
	file_name = "$(splitext(basename(@__FILE__))[1])_$(Dates.format(now(), "yyyymmdd_HHMMSS")).txt"
	open(file_name, "w") do f
		redirect_stdout(f) do
			redirect_stderr(f) do
				com_vb_gibbs()
			end	
		end
	end
end

out_txt()

# PLUTO_PROJECT_TOML_CONTENTS = """
# [deps]
# DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
# Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
# LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
# MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
# Optim = "429524aa-4258-5aef-a3af-852621145aeb"
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
# Optim = "~1.7.6"
# PDMats = "~0.11.17"
# Plots = "~1.38.16"
# PlutoUI = "~0.7.51"
# SpecialFunctions = "~2.2.0"
# StatsBase = "~0.34.0"
# StatsFuns = "~1.3.0"
# StatsPlots = "~0.15.5"
# """