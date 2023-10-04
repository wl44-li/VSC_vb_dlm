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
	using PDMats
	using StatsBase
	using Dates
end

# Gibbs sampling analog to VBEM-DLM
function sample_A(Xs, μ_A, Σ_A, Q)
    K, T = size(Xs)
	A = zeros(K, K)
	Σ_A_inv = inv(Σ_A)
	for i in 1:K
     	Σ_post = inv(Σ_A_inv + (Xs[:, 1:T-1] * Xs[:, 1:T-1]') ./ Q[i, i])
        μ_post = Σ_post * (Σ_A_inv * μ_A + (Xs[:, 1:T-1] * Xs[i, 2:T]) ./ Q[i, i])
		A[i, :] = rand(MvNormal(μ_post, Symmetric(Σ_post)))
    end
    return A
end

function sample_C(Xs, Ys, μ_C, Σ_C, R)
    P, T = size(Ys)
	K, _ = size(Xs)
    C_sampled = zeros(P, K)
	
    for i in 1:P
        Y = Ys[i, :]
        Σ_C_inv = inv(Σ_C)
        Σ_post = inv(Σ_C_inv + (Xs * Xs') ./ R[i, i])
        μ_post = Σ_post * (Σ_C_inv * μ_C + Xs * Y / R[i, i])
        C_sampled[i, :] = rand(MvNormal(μ_post, Symmetric(Σ_post)))
    end
    return C_sampled
end

function sample_R(Xs, Ys, C, α_ρ, β_ρ)
    P, T = size(Ys)
    ρ_sampled = zeros(P)
    for i in 1:P
        Y = Ys[i, :]
        α_post = α_ρ + T / 2
        β_post = β_ρ + 0.5 * sum((Y' - C[i, :]' * Xs).^2)
		
        ρ_sampled[i] = rand(Gamma(α_post, 1 / β_post))
    end
    return diagm(1 ./ ρ_sampled)
end

function ffbs_x(Ys, A, C, R, Q, μ_0, Σ_0)
	p, T = size(Ys)
    d, _ = size(A)
	
    # Initialize
    m = zeros(d, T)
	P = zeros(d, d, T)
	
	a = zeros(d, T)
	RR = zeros(d, d, T)
	X = zeros(d, T)

	a[:, 1] = A * μ_0
	P_1 = A * Σ_0 * A' + Q
	RR[:, :, 1] = P_1
	f_1 = C * a[:, 1]
    S_1 = C * P_1 * C' + R
    m[:, 1] = a[:, 1] + RR[:, :, 1] * C' * inv(S_1) * (Ys[:, 1] - f_1)
    P[:, :, 1] = RR[:, :, 1] - RR[:, :, 1] * C' * inv(S_1) * C * RR[:, :, 1]
		
		# Kalman filter (Prep 4.1)
    for t in 2:T
        # Prediction
        a[:, t] = A * m[:, t-1]
        P_t = A * P[:, :, t-1] * A' + Q
		RR[:, :, t] = P_t
		
		# Update
        f_t = C * a[:, t]
        S_t = C * P_t * C' + R

		# filter 
        m[:, t] = a[:, t] + RR[:, :, t] * C' * inv(S_t) * (Ys[:, t] - f_t)

       	Σ_t = RR[:, :, t] - RR[:, :, t] * C' * inv(S_t) * C * RR[:, :, t]
		P[:, :, t] = Σ_t
	end
	
		X[:, T] = rand(MvNormal(m[:, T], Symmetric(P[:, :, T])))
	
	# backward sampling
	for t in (T-1):-1:1
		h_t = m[:, t] +  P[:, :, t] * A' * inv(RR[:, :, t+1])*(X[:, t+1] - a[:, t+1])
		H_t = P[:, :, t] - P[:, :, t] * A' * inv(RR[:, :, t+1]) * A * P[:, :, t]
	
		X[:, t] = rand(MvNormal(h_t, Symmetric(H_t)))
	end

	# sample initial x_0
	h_0 = μ_0 + Σ_0 * A' * inv(RR[:, :, 1])*(X[:, 1] - a[:, 1])
	H_0 = Σ_0 - Σ_0 * A' * inv(RR[:, :, 1]) * A * Σ_0

	x_0 = rand((MvNormal(h_0, Symmetric(H_0))))

	return X, x_0
end

# Compare with Single-move
function sample_x_i(y_i, x_i_minus_1, x_i_plus_1, A, C, Q, R)
    Σ_x_i_inv = C' * inv(R) * C + inv(Q) + A' * inv(Q) * A
	Σ_x_i = inv(Σ_x_i_inv)
    μ_x_i = Σ_x_i * (C' * inv(R) * y_i + inv(Q) * A * x_i_minus_1 + A' * inv(Q) * x_i_plus_1)
	
    return rand(MvNormal(μ_x_i, Symmetric(Σ_x_i)))
end

function sample_x_1(y_1, x_2, A, C, Q, R)
    Σ_x_1_inv = C' * inv(R) * C + A' * inv(Q) * A
	Σ_x_1 = inv(Σ_x_1_inv)
    μ_x_1 = Σ_x_1 * (C' * inv(R) * y_1 + A' * inv(Q) * x_2)
    return rand(MvNormal(μ_x_1, Symmetric(Σ_x_1)))
end

function sample_x_T(y_T, x_T_1, A, C, Q, R)
    Σ_x_T_inv = C' * inv(R) * C + inv(Q)
	Σ_x_T = inv(Σ_x_T_inv)
    μ_x_T = Σ_x_T * (C' * inv(R) * y_T + inv(Q) * A * x_T_1)
    return rand(MvNormal(μ_x_T, Symmetric(Σ_x_T)))
end

function single_move_sampler(Ys, A, C, Q, R, mcmc=2000)
    p, T = size(Ys)
    d, _ = size(A)
	xs = rand(d, T)
    # Sample each latent state one at a time

	xss = zeros(d, T, mcmc)
	xss[:, :, 1] = xs
	
    for m in 2:mcmc
		xs[:, 1] = sample_x_1(Ys[:, 1], xs[:, 2], A, C, Q, R)

		for i in 2:T-2
			xs[:, i] = sample_x_i(Ys[:, i], xs[:, i-1], xs[:, i+1], A, C, Q, R)
		end
		xs[:, T] = sample_x_T(Ys[:, T], xs[:, T-1], A, C, Q, R)
		xss[:, :, m] = xs
    end
	
    return xss
end

function gibbs_dlm(y, K, single_move=false, Q=Matrix{Float64}(I, K, K), mcmc=3000, burn_in=1500, thinning=1)
	P, T = size(y)
	n_samples = Int.(mcmc/thinning)
    A_samples = zeros(K, K, n_samples)
    C_samples = zeros(P, K, n_samples)
    R_samples = zeros(P, P, n_samples)
    Xs_samples = zeros(K, T, n_samples)
	
    # Initialize A, C, and R, using fixed prior 
    A_init = rand(MvNormal(zeros(K), Matrix{Float64}(I, K, K)), K)'
    C_init = rand(MvNormal(zeros(K), Matrix{Float64}(I, K, K)), P)'
	a = 0.1
	b = 0.1
	ρ = rand(Gamma(a, b), 2)
    R_init = Diagonal(1 ./ ρ)
    A = A_init
    C = C_init
    R = R_init
	μ₀, Σ₀ = vec(mean(y, dims=2)), Matrix{Float64}(I, K, K)
	
    for iter in 1:(mcmc + burn_in)
		Xs = missing
        # Sample latent states Xs
		if single_move # not working well in Gibbs
			Xs = single_move_sampler(y, A, C, Q, R, 1)[:, :, 1]
		else
        	Xs, _ = ffbs_x(y, A, C, R, Q, μ₀, Σ₀)
		end
		
        # Sample model parameters A, C, and R
        A = sample_A(Xs, zeros(K), Σ₀, Q)
        C = sample_C(Xs, y, zeros(P), Σ₀, R)
        R = sample_R(Xs, y, C, a, b)
        # Store samples after burn-in
        if iter > burn_in && mod(iter - burn_in, thinning) == 0
			index = div(iter - burn_in, thinning)
            A_samples[:, :, index] = A
            C_samples[:, :, index] = C
            R_samples[:, :, index] = R
            Xs_samples[:, :, index] = Xs
        end
    end
    return A_samples, C_samples, R_samples, Xs_samples
end

# Multi-variate DLM with unknown $R, Q$
function sample_R_(y, x, C, v_0, S_0)
    T = size(y, 2)
	
    residuals = [y[:, t] - C * x[:, t] for t in 1:T]
	SS_y = sum([residuals[t] * residuals[t]' for t in 1:T])
	
    scale_posterior = S_0 + SS_y .* 0.5
    v_p = v_0 + 0.5 * T

	S_p = PDMat(Symmetric(inv(scale_posterior)))
	R⁻¹ = rand(Wishart(v_p, S_p))
    return inv(R⁻¹)
end

function sample_Q_(x, A, v_1, S_1, x_0)
    T = size(x, 2)
	
    residuals = [x[:, t] - A * x[:, t-1] for t in 2:T]
	SS_1 = sum([residuals[t] * residuals[t]' for t in 1:T-1])
    scale_posterior = S_1 + SS_1 .* 0.5

	scale_posterior += (x[:, 1] - A * x_0) * (x[:, 1] - A * x_0)' .* 0.5
    v_p = v_1 + 0.5 * T
	S_p = PDMat(Symmetric(inv(scale_posterior)))

	Q⁻¹ = rand(Wishart(v_p, S_p))
    return inv(Q⁻¹)
end

function gibbs_dlm_cov(y, A, C, mcmc=3000, burn_in=100, thinning=1)
	P, T = size(y)
	K = size(A, 2)
	
	μ_0 = vec(mean(y, dims=2)) 
	λ_0 = Matrix{Float64}(I, K, K)
	
	v_0 = P + 1.0 
	S_0 = Matrix{Float64}(0.01 * I, P, P)

	v_1 = K + 1.0
	S_1 = Matrix{Float64}(0.01 * I, K, K)
	
	# Initial values for the parameters
	R⁻¹ = rand(Wishart(v_0, inv(S_0)))
	Q⁻¹ = rand(Wishart(v_1, inv(S_1)))

	R, Q = inv(R⁻¹), inv(Q⁻¹)
	
	n_samples = Int.(mcmc/thinning)
	# Store the samples
	samples_X = zeros(n_samples, K, T)
	samples_Q = zeros(n_samples, K, K)
	samples_R = zeros(n_samples, P, P)
	
	# Gibbs sampler
	for i in 1:mcmc+burn_in
	    # Update the states
		x, x_0 = ffbs_x(y, A, C, R, Q, μ_0, λ_0)
		
		# Update the system noise
		Q = sample_Q_(x, A, v_1, S_1, x_0)
		
	    # Update the observation noise
		R = sample_R_(y, x, C, v_0, S_0)
	
	    # Store the samples
		if i > burn_in && mod(i - burn_in, thinning) == 0
			index = div(i - burn_in, thinning)
		    samples_X[index, :, :] = x
			samples_Q[index, :, :] = Q
		    samples_R[index, :, :] = R
		end
	end

	return samples_X, samples_Q, samples_R
end

# VBEM for general Q, R
begin
	struct Prior
	    ν_R::Float64
	    W_R::Array{Float64, 2}
	    ν_Q::Float64
	    W_Q::Array{Float64, 2}
	    μ_0::Array{Float64, 1}
	    Λ_0::Array{Float64, 2}
	end

	struct Q_Wishart
		ν_R_q
		W_R_q
		ν_Q_q
		W_Q_q
	end
end

function vb_m_step(y::Array{Float64, 2}, hss::HSS, prior::Prior, A::Array{Float64, 2}, C::Array{Float64, 2})
    _, T = size(y)
    
    # Compute the new parameters for the variational posterior of Λ_R
    ν_Rn = prior.ν_R + T
	W_Rn_inv = inv(prior.W_R) + y*y' - hss.S_C * C' - C * hss.S_C' + C * hss.W_C * C'
    
    # Compute the new parameters for the variational posterior of Λ_Q
    ν_Qn = prior.ν_Q + T
	W_Qn_inv = inv(prior.W_Q) + hss.W_C - hss.S_A * A' - A * hss.S_A' + A * hss.W_A * A'

	# Return expectations for E-step, Eq[R], E_q[Q], co-variance matrices
	return W_Rn_inv ./ ν_Rn, W_Qn_inv ./ ν_Qn, Q_Wishart(ν_Rn, inv(W_Rn_inv), ν_Qn, inv(W_Qn_inv))
	#return ν_Rn .* inv(W_Rn_inv), ν_Qn .* inv(W_Qn_inv) # precision matrices
end

function forward_(y::Array{Float64, 2}, A::Array{Float64, 2}, C::Array{Float64, 2}, E_R::Array{Float64, 2}, E_Q::Array{Float64, 2}, prior::Prior)
    P, T = size(y)
    K = size(A, 1)
    
    # Unpack the prior parameters
    μ_0, Λ_0 = prior.μ_0, prior.Λ_0
    
    # Initialize the filtered means and covariances
    μ_f = zeros(K, T)
    Σ_f = zeros(K, K, T)
    f_s = zeros(K, T)
	S_s = zeros(K, K, T)
    # Set the initial filtered mean and covariance to their prior values
	A_1 = A * μ_0
	R_1 = A * inv(Λ_0) * A' + E_Q
	
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
        Σ_p = A * Σ_f[:, :, t-1] * A' + E_Q

		# marginal y - normalization
		f_t = C * μ_p
		S_t = C * Σ_p * C' + E_R
		f_s[:, t] = f_t
		S_s[:, :, t] = S_t
		
		# Filtered state mean and covariance (2.8a - 2.8c DLM with R)
		μ_f[:, t] = μ_p + Σ_p * C' * inv(S_t) * (y[:, t] - f_t)
		Σ_f[:, :, t] = Σ_p - Σ_p * C' * inv(S_t) * C * Σ_p
			
		# Kalman gain
        #K_t = Σ_p * C' / (C * Σ_p * C' + E_R)
        #μ_f[:, t] = μ_p + K_t * (y[:, t] - C * μ_p)
        #Σ_f[:, :, t] = (I - K_t * C) * Σ_p
    end
	
    log_z = sum(logpdf(MvNormal(f_s[:, i], Symmetric(S_s[:, :, i])), y[:, i]) for i in 1:T)
    return μ_f, Σ_f, log_z
end

function backward_(μ_f::Array{Float64, 2}, Σ_f::Array{Float64, 3}, A::Array{Float64, 2}, E_Q::Array{Float64, 2})
    K, T = size(μ_f)
    
    # Initialize the smoothed means, covariances, and cross-covariances
    μ_s = zeros(K, T)
    Σ_s = zeros(K, K, T)
    Σ_s_cross = zeros(K, K, T)
    
    # Set the final smoothed mean and covariance to their filtered values
    μ_s[:, T] = μ_f[:, T]
    Σ_s[:, :, T] = Σ_f[:, :, T]
    
    # Backward pass
    for t = T-1:-1:1
        # Compute the gain J_t
        J_t = Σ_f[:, :, t] * A' / (A * Σ_f[:, :, t] * A' + E_Q)

        # Update the smoothed mean μ_s and covariance Σ_s
        μ_s[:, t] = μ_f[:, t] + J_t * (μ_s[:, t+1] - A * μ_f[:, t])
        Σ_s[:, :, t] = Σ_f[:, :, t] + J_t * (Σ_s[:, :, t+1] - A * Σ_f[:, :, t] * A' - E_Q) * J_t'

        # Compute the cross covariance Σ_s_cross
        #Σ_s_cross[:, :, t+1] = inv(inv(Σ_f[:, :, t]) + A'*A) * A' * Σ_s[:, :, t+1]
		Σ_s_cross[:, :, t+1] = J_t * Σ_s[:, :, t+1]
    end
	
	Σ_s_cross[:, :, 1] = inv(I + A'*A) * A' * Σ_s[:, :, 1]
	#J_1 = I * A' / (A * I * A' + E_Q)
	#Σ_s_cross[:, :, 1] = J_1 * Σ_s[:, :, 1]
    return μ_s, Σ_s, Σ_s_cross
end

function vb_e_step(y::Array{Float64, 2}, A::Array{Float64, 2}, C::Array{Float64, 2}, E_R::Array{Float64, 2}, E_Q::Array{Float64, 2}, prior::Prior)
    # Run the forward pass
    μ_f, Σ_f, log_Z = forward_(y, A, C, E_R, E_Q, prior)

    # Run the backward pass
    μ_s, Σ_s, Σ_s_cross = backward_(μ_f, Σ_f, A, E_Q)

    # Compute the hidden state sufficient statistics
    W_C = sum(Σ_s, dims=3)[:, :, 1] + μ_s * μ_s'
    W_A = sum(Σ_s[:, :, 1:end-1], dims=3)[:, :, 1] + μ_s[:, 1:end-1] * μ_s[:, 1:end-1]'
    #S_C = y * μ_s'
	S_C = μ_s * y'
    S_A = sum(Σ_s_cross, dims=3)[:, :, 1] + μ_s[:, 1:end-1] * μ_s[:, 2:end]'
    W_Y = y * y'

	# Return the hidden state sufficient statistics
    return HSS(W_C, W_A, S_C, S_A), log_Z
end

function vbem_(y::Array{Float64, 2}, A::Array{Float64, 2}, C::Array{Float64, 2}, prior::Prior, max_iter=300)
	
	hss = HSS(ones(size(A)), ones(size(A)), ones(size(C)), ones(size(A)))
	E_R, E_Q  = missing, missing
	
	for i in 1:max_iter
		E_R, E_Q, _ = vb_m_step(y, hss, prior, A, C)
				
		hss, _ = vb_e_step(y, A, C, E_R, E_Q, prior)
	end

	return E_R, E_Q
end

function vbem_his_plot(y::Array{Float64, 2}, A::Array{Float64, 2}, C::Array{Float64, 2}, prior::Prior, max_iter=100)
    P, T = size(y)
    K, _ = size(A)

    W_C = zeros(K, K)
    W_A = zeros(K, K)
    S_C = zeros(P, K)
    S_A = zeros(K, K)
    hss = HSS(W_C, W_A, S_C, S_A)
	E_R, E_Q  = missing, missing
	
    # Initialize the history of E_R and E_Q
    E_R_history = zeros(P, P, max_iter)
    E_Q_history = zeros(K, K, max_iter)

    # Repeat until convergence
    for iter in 1:max_iter
		E_R, E_Q, _ = vb_m_step(y, hss, prior, A, C)
				
		hss, _ = vb_e_step(y, A, C, E_R, E_Q, prior)

        # Store the history of E_R and E_Q
        E_R_history[:, :, iter] = E_R
        E_Q_history[:, :, iter] = E_Q
    end

	p1 = plot(title = "Learning of R")

	# Show progress of diagonal entries
    for i in 1:P
        plot!(10:max_iter, [E_R_history[i, i, t] for t in 10:max_iter], label = "R[$i, $i]")
    end

    p2 = plot(title = "Learning of Q")
    for i in 1:K
        plot!(10:max_iter, [E_Q_history[i, i, t] for t in 10:max_iter], label = "Q[$i, $i]")
    end
	
	function_name = "R_Q_Progress"
	p = plot(p1, p2, layout = (1, 2))
	plot_file_name = "$(splitext(basename(@__FILE__))[1])_$(function_name)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).svg"
    savefig(p, joinpath(expanduser("~/Downloads/_graphs"), plot_file_name))
end

# With Convergence Check
function vbem_c(y::Array{Float64, 2}, A::Array{Float64, 2}, C::Array{Float64, 2}, prior::Prior, max_iter=200, tol=5e-3)
	hss = HSS(ones(size(A)), ones(size(A)), ones(size(C)), ones(size(A)))
	E_R, E_Q  = missing, missing
	elbo_prev = -Inf
	el_s = zeros(max_iter)
	for i in 1:max_iter
		E_R, E_Q, Q_Wi = vb_m_step(y, hss, prior, A, C)
				
		hss, log_Z = vb_e_step(y, A, C, E_R, E_Q, prior)

		kl_Wi = kl_Wishart(Q_Wi.ν_R_q, Q_Wi.W_R_q, prior.ν_R, prior.W_R) + kl_Wishart(Q_Wi.ν_Q_q, Q_Wi.W_Q_q, prior.ν_Q, prior.W_Q)
		elbo = log_Z - kl_Wi
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

function test_vb()
	A = [1.0 0.0; 0.0 1.0]
	C = [1.0 0.0; 0.0 1.0]
	Q = Diagonal([1.0, 1.0])
	R = Diagonal([0.1, 0.1])
	T = 500
	Random.seed!(133)
	μ_0 = [0.0, 0.0]
	Σ_0 = Diagonal([1.0, 1.0])
	y, x_true = gen_data(A, C, Q, R, μ_0, Σ_0, T)
	D, _ = size(y)
	K = size(A, 1)
	W_Q = Matrix{Float64}(100*I, K, K)
	W_R = Matrix{Float64}(100*I, D, D)
	prior = Prior(D + 1.0, W_R, K + 1.0, W_Q, zeros(K), Matrix{Float64}(I, K, K))
	
	vbem_his_plot(y, A, C, prior, 100)

	println("--- VB ---")
	@time R, Q, elbos = vbem_c(y, A, C, prior)
	p = plot(elbos, label = "elbo", title = "ElBO progression")
	display(p)
	plot_file_name = "$(splitext(basename(@__FILE__))[1])_$(Dates.format(now(), "yyyymmdd_HHMMSS")).svg"
	savefig(p, joinpath(expanduser("~/Downloads/_graphs"), plot_file_name))
	μs_f, σs_f2 = forward_(y, A, C, R, Q, prior)
    μs_s, _, _ = backward_(μs_f, σs_f2, A, Q)
	println("MSE, MAD of VB: ", error_metrics(x_true, μs_s))

	println("--- MCMC ---")
	@time Xs_samples, Qs_samples, Rs_samples = gibbs_dlm_cov(y, A, C, 3000, 1000, 1)
	Q_chain = Chains(reshape(Qs_samples, 3000, 4))
	R_chain = Chains(reshape(Rs_samples, 3000, 4))

	summary_stats_q = summarystats(Q_chain)
	summary_stats_r = summarystats(R_chain)
	summary_df_q = DataFrame(summary_stats_q)
	summary_df_r = DataFrame(summary_stats_r)
	println("Q summary stats: ", summary_df_q)
	println("R summary stats: ", summary_df_r)

	xs_m = mean(Xs_samples, dims=1)[1, :, :]
	println("MSE, MAD of MCMC X mean: ", error_metrics(x_true, xs_m))
	println("MSE, MAD of MCMC X end: ", error_metrics(x_true, Xs_samples[end, :, :]))
end

test_vb()

# Restrict R, Q as diagonal matrices
function vb_m_diag(y, hss::HSS, hpp::HPP_D, A::Array{Float64, 2}, C::Array{Float64, 2})
    D, T = size(y)
    K = size(A, 1)
    # Compute the new parameters for the variational posterior of Λ_R
	G = y*y' - hss.S_C * C' - C * hss.S_C' + C * hss.W_C * C'
    a_ = hpp.a + 0.5 * T
	a_s = a_ * ones(D)
    b_s = [hpp.b + 0.5 * G[i, i] for i in 1:D]
	q_ρ = Gamma.(a_s, 1 ./ b_s)
	Exp_R⁻¹ = diagm(mean.(q_ρ))
	
    # Compute the new parameters for the variational posterior of Λ_Q
    H = hss.W_C - hss.S_A * A' - A * hss.S_A' + A * hss.W_A * A'
	α_ = hpp.α + 0.5 * T
	α_s = α_ * ones(K)
    β_s = [hpp.β + 0.5 * H[i, i] for i in 1:K]
	q_𝛐 = Gamma.(α_s, 1 ./ β_s)	
	Exp_Q⁻¹= diagm(mean.(q_𝛐))
	
	return Exp_R⁻¹, Exp_Q⁻¹, Q_Gamma(a_, b_s, α_, β_s)
end

function forward_v(y, A::Array{Float64, 2}, C::Array{Float64, 2}, E_R, E_Q, prior::HPP_D)
    P, T = size(y)
    K = size(A, 1)
    
    # Unpack the prior parameters
    μ_0, Λ_0 = prior.μ_0, prior.Σ_0
    
    # Initialize the filtered means and covariances
    μ_f = zeros(K, T)
    Σ_f = zeros(K, K, T)

    f_s = zeros(P, T)
	S_s = zeros(P, P, T)
	
    # Set the initial filtered mean and covariance to their prior values
	A_1 = A * μ_0
	R_1 = A * inv(Λ_0) * A' + E_Q
	
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
        Σ_p = A * Σ_f[:, :, t-1] * A' + E_Q

		# marginal y - normalization
		f_t = C * μ_p
		S_t = C * Σ_p * C' + E_R
		f_s[:, t] = f_t
		S_s[:, :, t] = S_t
		
		# Filtered state mean and covariance (2.8a - 2.8c DLM with R)
		μ_f[:, t] = μ_p + Σ_p * C' * inv(S_t) * (y[:, t] - f_t)
		Σ_f[:, :, t] = Σ_p - Σ_p * C' * inv(S_t) * C * Σ_p
			
		# Kalman gain
        #K_t = Σ_p * C' / (C * Σ_p * C' + E_R)
        #μ_f[:, t] = μ_p + K_t * (y[:, t] - C * μ_p)
        #Σ_f[:, :, t] = (I - K_t * C) * Σ_p
    end
	
    log_z = sum(logpdf(MvNormal(f_s[:, i], Symmetric(S_s[:, :, i])), y[:, i]) for i in 1:T)
    return μ_f, Σ_f, log_z
end

function backward_v(μ_f::Array{Float64, 2}, Σ_f::Array{Float64, 3}, A::Array{Float64, 2}, E_Q::Array{Float64, 2}, prior)
    K, T = size(μ_f)
    
    # Initialize the smoothed means, covariances, and cross-covariances
    μ_s = zeros(K, T)
    Σ_s = zeros(K, K, T)
    Σ_s_cross = zeros(K, K, T)
    
    # Set the final smoothed mean and covariance to their filtered values
    μ_s[:, T] = μ_f[:, T]
    Σ_s[:, :, T] = Σ_f[:, :, T]
    
    # Backward pass
    for t = T-1:-1:1
        # Compute the gain J_t
        J_t = Σ_f[:, :, t] * A' / (A * Σ_f[:, :, t] * A' + E_Q)

        # Update the smoothed mean μ_s and covariance Σ_s
        μ_s[:, t] = μ_f[:, t] + J_t * (μ_s[:, t+1] - A * μ_f[:, t])
        Σ_s[:, :, t] = Σ_f[:, :, t] + J_t * (Σ_s[:, :, t+1] - A * Σ_f[:, :, t] * A' - E_Q) * J_t'

        # Compute the cross covariance Σ_s_cross
        #Σ_s_cross[:, :, t+1] = inv(inv(Σ_f[:, :, t]) + A'*A) * A' * Σ_s[:, :, t+1]
		Σ_s_cross[:, :, t+1] = J_t * Σ_s[:, :, t+1]
    end
	
	Σ_s_cross[:, :, 1] = inv(I + A'*A) * A' * Σ_s[:, :, 1]
	
	J_0 = prior.Σ_0 * A' / (A * prior.Σ_0 * A' + E_Q)
	μ_s0 = prior.μ_0 + J_0 * (μ_s[:, 1] -  A * prior.μ_0)
	Σ_s0 = prior.Σ_0 + J_0 * (Σ_s[:, :, 1] - A * prior.Σ_0 * A' - E_Q) * J_0'
    return μ_s, Σ_s, μ_s0, Σ_s0, Σ_s_cross
end

function vb_e_diag(y, A::Array{Float64, 2}, C::Array{Float64, 2}, E_R, E_Q, prior)
    # Run the forward pass
    μ_f, Σ_f, log_Z = forward_v(y, A, C, E_R, E_Q, prior)

    # Run the backward pass
    μ_s, Σ_s, μ_s0, Σ_s0, Σ_s_cross = backward_v(μ_f, Σ_f, A, E_Q, prior)

    # Compute the hidden state sufficient statistics
    W_C = sum(Σ_s, dims=3)[:, :, 1] + μ_s * μ_s'
    W_A = sum(Σ_s[:, :, 1:end-1], dims=3)[:, :, 1] + μ_s[:, 1:end-1] * μ_s[:, 1:end-1]'
	W_A += Σ_s0 + μ_s0*μ_s0'
	
    S_C = μ_s * y'
    S_A = sum(Σ_s_cross, dims=3)[:, :, 1] + μ_s[:, 1:end-1] * μ_s[:, 2:end]'
	S_A += μ_s0*μ_s[:, 1]'
    W_Y = y * y'

	# Return the hidden state sufficient statistics
    return HSS(W_C, W_A, S_C, S_A), μ_s0, Σ_s0, log_Z
end

# VBEM with Convergence
function vbem_c_diag(y, A::Array{Float64, 2}, C::Array{Float64, 2}, prior, hp_learn=false, max_iter=200, tol=5e-3)

	D, _ = size(y)
	K = size(A, 1)
	hss = HSS(ones(size(A)), ones(size(A)), ones(size(C)), ones(size(A)))
	E_R_inv, E_Q_inv = missing, missing
	elbo_prev = -Inf
	el_s = zeros(max_iter)
	
	
	for i in 1:max_iter
		E_R_inv, E_Q_inv, Q_gam = vb_m_diag(y, hss, prior, A, C)
		hss, μ_s0, Σ_s0, log_Z = vb_e_diag(y, A, C, inv(E_R_inv), inv(E_Q_inv), prior)
		
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
	# A, C identity matrix (cf. local level model)
	A = [1.0 0.0; 0.0 1.0]
	C = [1.0 0.0; 0.0 1.0]
	Q = Diagonal([1.0, 1.0])
	R = Diagonal([0.1, 0.1])
	T = 500
	Random.seed!(1)
	μ_0 = [0.0, 0.0]
	Σ_0 = Diagonal([1.0, 1.0])
	y, x_true = gen_data(A, C, Q, R, μ_0, Σ_0, T)
	K = size(A, 1)
	prior = HPP_D(0.01, 0.01, 0.01, 0.01, zeros(K), Matrix{Float64}(I, K, K))
	println("\n--- VB with Diagonal Covariances ---")

	for t in [false, true]
		println("\nHyperparam optimisation: $t")
		@time R, Q, elbos = vbem_c_diag(y, A, C, prior, t)
		p = plot(elbos, label = "elbo", title = "ElBO Progression, Hyperparam optim: $t")
		#display(p)
		plot_file_name = "$(splitext(basename(@__FILE__))[1])_$(Dates.format(now(), "yyyymmdd_HHMMSS")).svg"
		savefig(p, joinpath(expanduser("~/Downloads/_graphs"), plot_file_name))

		μs_f, σs_f2 = forward_v(y, A, C, R, Q, prior)
		μs_s, _, _ = backward_v(μs_f, σs_f2, A, Q, prior)
		println("MSE, MAD of VB X: ", error_metrics(x_true, μs_s))
		sleep(1)
	end
end

main()

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