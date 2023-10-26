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

function forward_filter(Ys, A, C, R, Q, m_0, C_0)
	_, T = size(Ys)
	d, _ = size(A)
	
	# Initialize, using "DLM with R" notation
	ms = zeros(d, T+1)
	Cs = zeros(d, d, T+1)

	ms[:, 1] = m_0
	Cs[:, :, 1] = C_0
	
	# one-step ahead latent distribution, used in backward sampling
	as = zeros(d, T)
	Rs = zeros(d, d, T)

	for t in 1:T
		# Prediction
		as[:, t] = a_t = A * ms[:, t]
		Rs[:, :, t] = R_t = A * Cs[:, :, t] * A' + Q #W
		
		# Update
		f_t = C * a_t
		S_t = C * R_t * C' + R #V

		# filter 
		ms[:, t+1] = a_t + R_t * C' * inv(S_t) * (Ys[:, t] - f_t)
		Cs[:, :, t+1]= R_t - R_t * C' * inv(S_t) * C * R_t
	end
	return ms, Cs, as, Rs
end


function ffbs_x(Ys, A, C, R, Q, m_0, P_0)
	d, _ = size(A)
	_, T = size(Ys)

	ms, Cs, as, Rs = forward_filter(Ys, A, C, R, Q, m_0, P_0)
	X = zeros(d, T+1)

	# DEBUG FFBS
	try
		X[:, end] = rand(MvNormal(ms[:, end], Symmetric(Cs[:, :, end])))
	catch PosDefException
		println("Pathology case encountered: ")
		println("C_end: ")
		println(Cs[:, :, end])
		println("Y_end:")
		println(Ys[:, end])
	end

	# backward sampling
	for t in T:-1:1
		h_t = ms[:, t] + Cs[:, :, t] * A' * inv(Rs[:, :, t])*(X[:, t+1] - as[:, t])
		H_t = Cs[:, :, t] - Cs[:, :, t] * A' * inv(Rs[:, :, t]) * A * Cs[:, :, t]
		X[:, t] = rand(MvNormal(h_t, Symmetric(H_t)))
	end
	return X
end


# Multi-variate DLM with full unknown $R, Q$
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

# not-yet used 
# function sample_Q_(x, A, v_1, S_1, x_0)
#     T = size(x, 2)
	
#     residuals = [x[:, t] - A * x[:, t-1] for t in 2:T]
# 	SS_1 = sum([residuals[t] * residuals[t]' for t in 1:T-1])
#     scale_posterior = S_1 + SS_1 .* 0.5

# 	scale_posterior += (x[:, 1] - A * x_0) * (x[:, 1] - A * x_0)' .* 0.5
#     v_p = v_1 + 0.5 * T
# 	S_p = PDMat(Symmetric(inv(scale_posterior)))

# 	Q⁻¹ = rand(Wishart(v_p, S_p))
#     return inv(Q⁻¹)
# end

function gibbs_dlm_cov(y, A, C, mcmc=3000, burn_in=1500, thinning=1, debug=false)
	P, T = size(y)
	K = size(A, 2)
	
	μ_0 = vec(mean(y, dims=2)) 
	λ_0 = Matrix{Float64}(I, K, K)
	
	v_0 = P + 1.0 
	S_0 = Matrix{Float64}(0.01 * I, P, P)

	# Initial values for the parameters
	R⁻¹ = rand(Wishart(v_0, inv(S_0)))
	R = inv(R⁻¹)

	# DEBUG: hyper-prior [Pathological Gamma(0.01, 0.01)], (0.1, 0.1)?
	α, β = 100, 100
	ρ_q = rand(Gamma(α, β), K)
    Q = Diagonal(1 ./ ρ_q)
	
	n_samples = Int.(mcmc/thinning)
	# Store the samples
	samples_X = zeros(n_samples, K, T)
	samples_Q = zeros(n_samples, K, K)
	samples_R = zeros(n_samples, P, P)
	
	# Gibbs sampler
	for i in 1:mcmc+burn_in
	    # Update the states

		if debug
			println("MCMC full R debug, iteration: $i")
		end
		x = ffbs_x(y, A, C, R, Q, μ_0, λ_0)
		x = x[:, 2:end]

		# Update the system noise
		Q = sample_Q(x, A, α, β)

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
    _, T = size(y)
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
function vbem_c(y::Array{Float64, 2}, A::Array{Float64, 2}, C::Array{Float64, 2}, prior::Prior, max_iter=1000, tol=5e-3)
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

	# Return the hidden state sufficient statistics
    return HSS(W_C, W_A, S_C, S_A), μ_s0, Σ_s0, log_Z
end

# VBEM with Convergence
function vbem_c_diag(y, A::Array{Float64, 2}, C::Array{Float64, 2}, prior, hp_learn=false, max_iter=500, tol=1e-3)

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

# Gibbs sampling of diagonal co-variances
function sample_R(Xs, Ys, C, a_ρ, b_ρ)
    P, T = size(Ys)
    ρ_sampled = zeros(P)
    for i in 1:P
        Y = Ys[i, :]
        a_post = a_ρ + T / 2
        b_post = b_ρ + 0.5 * sum((Y' - C[i, :]' * Xs).^2)
		
        ρ_sampled[i] = rand(Gamma(a_post, 1 / b_post))
    end
    return diagm(1 ./ ρ_sampled)
end

function sample_Q(Xs, A, α_q, β_q)
    K, T = size(Xs)
    q_sampled = zeros(K)
    for i in 1:K
        X_diff = Xs[i, 2:end] - (A * Xs[:, 1:end-1])[i, :]
        α_post = α_q + T / 2 - 1  # Subtracting 1 as the first state doesn't have a predecessor
        β_post = β_q + 0.5 * sum(X_diff.^2)
        
        q_sampled[i] = rand(Gamma(α_post, 1 / β_post))
    end
    return diagm(1 ./ q_sampled)
end

function test_Gibbs_RQ()
	A = [1.0 0.0; 0.0 1.0]
	C = [1.0 0.0; 0.0 1.0]
	Q = Diagonal([1.0, 1.0])
	R = Diagonal([0.1, 0.1])
	μ_0 = [0.0, 0.0]
	Σ_0 = Diagonal([1.0, 1.0])
	Random.seed!(123)
	T = 1000
	y, x_true = gen_data(A, C, Q, R, μ_0, Σ_0, T)
	prior = Q_Gamma(100, 100, 100, 100)

	R = sample_R(x_true, y, C, prior.a, prior.b)
	Q = sample_Q(x_true, A, prior.α, prior.β)

	println("R:")
	show(stdout, "text/plain", R)
	println()
	println("Q:")
	show(stdout, "text/plain", Q)
end

function gibbs_diag(y, A, C, prior::HPP_D, mcmc=3000, burn_in=1500, thinning=1, debug=false)
	P, T = size(y)
	K = size(A, 2)
	
	m_0 = prior.μ_0
	P_0 = prior.Σ_0
	
	a, b, α, β = prior.a, prior.b, prior.α, prior.β
	ρ_r = rand(Gamma(a, b), P)
    R = Diagonal(1 ./ ρ_r)
	ρ_q = rand(Gamma(α, β), K)
    Q = Diagonal(1 ./ ρ_q)

	n_samples = Int.(mcmc/thinning)
	# Store the samples
	samples_X = zeros(n_samples, K, T)
	samples_Q = zeros(n_samples, K, K)
	samples_R = zeros(n_samples, P, P)
	
	# Gibbs sampler
	for i in 1:mcmc+burn_in
		if debug
			println("MCMC diagonal R, Q debug, iteration : $i")
		end
	    # Update the states
		x = ffbs_x(y, A, C, R, Q, m_0, P_0)
		x = x[:, 2:end]

		# Update the system noise
		Q = sample_Q(x, A, α, β)
		
	    # Update the observation noise
		R = sample_R(x, y, C, a, b)
	
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

function test_gibbs_diag(y, x_true, mcmc=10000, burn_in=5000, thin=1)
	A = [1.0 0.0; 0.0 1.0]
	C = [1.0 0.0; 0.0 1.0]
	K = size(A, 1)
	D, _ = size(y)
	prior = HPP_D(100, 100, 100, 100, zeros(K), Matrix{Float64}(I, K, K))
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
	println("\nMSE, MAD of MCMC X mean: ", error_metrics(x_true, xs_m))
end

function test_vb(y, x_true)
	A = [1.0 0.0; 0.0 1.0]
	C = [1.0 0.0; 0.0 1.0]
	K = size(A, 1)
	prior = HPP_D(0.01, 0.01, 0.01, 0.01, zeros(K), Matrix{Float64}(I, K, K))
	println("\n--- VB with Diagonal Covariances ---")

	for t in [false, true]
		println("\nHyperparam optimisation: $t")
		@time R, Q, elbos = vbem_c_diag(y, A, C, prior, t)
		#p = plot(elbos, label = "elbo", title = "ElBO Progression, Hyperparam optim: $t")
		#display(p)
		#plot_file_name = "$(splitext(basename(@__FILE__))[1])_$(Dates.format(now(), "yyyymmdd_HHMMSS")).svg"
		#savefig(p, joinpath(expanduser("~/Downloads/_graphs"), plot_file_name))
		println("VB q(R): ")
		show(stdout, "text/plain", R)
		println("\n\n VB q(Q): ")
		show(stdout, "text/plain", Q)
		μs_f, σs_f2 = forward_v(y, A, C, R, Q, prior)
		μs_s, _, _ = backward_v(μs_f, σs_f2, A, Q, prior)
		println("\n\nMSE, MAD of VB X: ", error_metrics(x_true, μs_s))
		sleep(1)
	end

	D, _ = size(y)
	W_Q = Matrix{Float64}(I, K, K)
	W_R = Matrix{Float64}(I, D, D)
	prior = Prior(D + 1.0, W_R, K + 1.0, W_Q, zeros(K), Matrix{Float64}(I, K, K))
	println("\n--- VB with Full Co-variances ---")
	@time R, Q, elbos = vbem_c(y, A, C, prior)
	println("VB q(R): ")
	show(stdout, "text/plain", R)
	println("\n\nVB q(Q): ")
	show(stdout, "text/plain", Q)
	#p = plot(elbos, label = "elbo", title = "ElBO progression, seed = $rnd")
	#display(p)
	#plot_file_name = "$(splitext(basename(@__FILE__))[1])_$(Dates.format(now(), "yyyymmdd_HHMMSS")).svg"
	#savefig(p, joinpath(expanduser("~/Downloads/_graphs"), plot_file_name))
	μs_f, σs_f2 = forward_(y, A, C, R, Q, prior)
    μs_s, _, _ = backward_(μs_f, σs_f2, A, Q)
	println("\n\nMSE, MAD of VB X: ", error_metrics(x_true, μs_s))
end

function test_gibbs_cov(y, x_true, mcmc=10000, burn_in=5000, thin=1)
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
	println("MSE, MAD of MCMC X mean: ", error_metrics(x_true, xs_m))
	#println("MSE, MAD of MCMC X end: ", error_metrics(x_true, Xs_samples[end, :, :]))
end

function test_data(rnd, max_T = 500)
	A = [1.0 0.0; 0.0 1.0]
	C = [1.0 0.0; 0.0 1.0]
	Q = Diagonal([1.0, 1.0]) # Diagonal Q
	R = [0.5 0.2; 0.2 0.5] # full-cov R
	μ_0 = [0.0, 0.0]
	Σ_0 = Diagonal([1.0, 1.0])
	Random.seed!(rnd)
	y, x_true = gen_data(A, C, Q, R, μ_0, Σ_0, max_T)
	return y, x_true
end

function test_ffbs()
	y, x_true = test_data(1)
	A = [1.0 0.0; 0.0 1.0]
	C = [1.0 0.0; 0.0 1.0]
	Q = Diagonal([1.0, 1.0])
	R = [0.5 0.2; 0.2 0.5]
	m_0 = [0.0, 0.0]
	P_0 = Diagonal([1.0, 1.0])
	_, Ps, _, _ = forward_filter(y, A, C, R, Q, m_0, P_0)
	println(Ps[:, :, end])
	X = ffbs_x(y, A, C, R, Q, m_0, P_0)

	X = X[:, 2:end]
	println("MSE, MAD of FFBS: ", error_metrics(x_true, X))
	plot_latent(x_true', X')
end

#test_ffbs()

function test_gibbs()
	seeds = [92, 134, 103, 133, 233]
	#108, 123, 
	for sd in seeds
		y, x_true = test_data(sd)
		println("--- Seed: $sd ---")
		test_gibbs_cov(y, x_true, 10000, 5000, 1)
		println()
		test_gibbs_diag(y, x_true, 10000, 5000, 1) # Debug: to investigate
	end
end

#test_gibbs()

function com_vb_gibbs()
	seeds = [108, 134, 123, 105, 233]
	#188, 199, 233, 234, 236
	for sd in seeds
		y, x_true = test_data(sd)
		println("--- Seed: $sd ---")
		test_gibbs_cov(y, x_true, 20000, 10000, 1)
		println()
		test_vb(y, x_true)
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