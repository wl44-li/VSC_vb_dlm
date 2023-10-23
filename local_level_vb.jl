include("kl_optim.jl")

begin
	using Distributions, Random
	using LinearAlgebra
	using SpecialFunctions
	using StateSpaceModels
	using MCMCChains
	using DataFrames
	using StatsPlots
end

begin
	struct HPP_uni
	    a::Float64
	    b::Float64
	    α::Float64
	    γ::Float64
		μ₀::Float64
		σ₀::Float64
	end
	
	struct HSS_uni
	    W_A::Float64
	    S_A::Float64
	    W_C::Float64
	    S_C::Float64
	end

	struct Exp_ϕ_uni
		A
		C
		R⁻¹
		AᵀA
		CᵀR⁻¹C
		R⁻¹C
		CᵀR⁻¹
	end
end

function vb_m_uni(y::Vector{Float64}, hss::HSS_uni, hpp::HPP_uni)
	T = length(y)
    a, b, α, γ, μ_0, σ_0 = hpp.a, hpp.b, hpp.α, hpp.γ, hpp.μ₀, hpp.σ₀
	W_A, S_A, W_C, S_C = hss.W_A, hss.S_A, hss.W_C, hss.S_C

	σ_A = 1 / (α + W_A)
    σ_C = 1 / (γ + W_C)	
	G = y' * y - S_C * σ_C * S_C

	# Update parameters of Gamma distribution
	a_n = a + 0.5 * T
	b_n = b + 0.5 * G

	q_ρ = Gamma(a_n, 1 / b_n)
	ρ̄ = mean(q_ρ)

	Exp_A = S_A*σ_A
	Exp_C = S_C*σ_C
	Exp_R⁻¹ = ρ̄

	Exp_AᵀA = Exp_A^2 + σ_A
    Exp_CᵀR⁻¹C = Exp_C^2 * Exp_R⁻¹ + σ_C
    Exp_R⁻¹C = Exp_C * Exp_R⁻¹
    Exp_CᵀR⁻¹ = Exp_R⁻¹C 

	return Exp_ϕ_uni(Exp_A, Exp_C, Exp_R⁻¹, Exp_AᵀA, Exp_CᵀR⁻¹C, Exp_R⁻¹C, Exp_CᵀR⁻¹)
end

function v_forward(y::Vector{Float64}, exp_np::Exp_ϕ_uni, μ_0, σ_0)
	T = length(y)

    μs = zeros(T)
    σs = zeros(T)
	σs_ = zeros(T)
	
	# TO-DO: ELBO and convergence check
	#Qs = zeros(D, D, T)
	#fs = zeros(D, T)

	# initialise for t=1
	σ₀_ = 1 / (σ_0^(-1) + exp_np.AᵀA)
	σs_[1] = σ₀_
	
    σs[1] = 1/ (1.0 + exp_np.CᵀR⁻¹C - exp_np.A*σ₀_*exp_np.A)
    μs[1] = σs[1]*(exp_np.CᵀR⁻¹*y[1] + exp_np.A*σ₀_*σ_0^(-1)*μ_0)

	# iterate over T
	for t in 2:T
		σₜ₋₁_ = 1/ ((σs[t-1])^(-1) + exp_np.AᵀA)
		σs_[t] = σₜ₋₁_
		
		σs[t] = 1/ (1.0 + exp_np.CᵀR⁻¹C - exp_np.A*σₜ₋₁_*exp_np.A)
    	μs[t] = σs[t]*(exp_np.CᵀR⁻¹*y[t] + exp_np.A*σₜ₋₁_*(σs[t-1])^(-1)*μs[t-1])

	end

	return μs, σs, σs_
end

function v_backward(y::Vector{Float64}, exp_np::Exp_ϕ_uni)
	T = length(y)
	ηs = zeros(T)
    Ψs = zeros(T)

    # Initialize the filter, t=T, β(x_T-1)
	Ψs[T] = 0.0
    ηs[T] = 1.0
	
	Ψₜ = 1/(1.0 + exp_np.CᵀR⁻¹C)
	Ψs[T-1] = 1/ (exp_np.AᵀA - exp_np.A*Ψₜ*exp_np.A)
	ηs[T-1] = Ψs[T-1]*exp_np.A*Ψₜ*exp_np.CᵀR⁻¹*y[T]
	
	for t in T-2:-1:1
		Ψₜ₊₁ = 1/(1.0 + exp_np.CᵀR⁻¹C + (Ψs[t+1])^(-1))
		
		Ψs[t] = 1/ (exp_np.AᵀA - exp_np.A*Ψₜ₊₁*exp_np.A)
		ηs[t] = Ψs[t]*exp_np.A*Ψₜ₊₁*(exp_np.CᵀR⁻¹*y[t+1] + (Ψs[t+1])^(-1)*ηs[t+1])
	end

	# for t=1, this correspond to β(x_0), the probability of all the data given the setting of the auxiliary x_0 hidden state.
	Ψ₁ = 1/ (1.0 + exp_np.CᵀR⁻¹C + (Ψs[1])^(-1))
	Ψ_0 = 1/ (exp_np.AᵀA - exp_np.A*Ψ₁*exp_np.A)
	η_0 = Ψs[1]*exp_np.A*Ψ₁*(exp_np.CᵀR⁻¹*y[1] + (Ψs[1])^(-1)*ηs[1])
	
	return ηs, Ψs, η_0, Ψ_0
end

function parallel_smoother(μs, σs, ηs, Ψs, η_0, Ψ_0, μ_0, σ_0)
	T = length(μs)
	Υs = zeros(T)
	ωs = zeros(T)

	# ending condition t=T
	Υs[T] = σs[T]
	ωs[T] = μs[T]
	
	for t in 1:(T-1)
		Υs[t] = 1 / ((σs[t])^(-1) + (Ψs[t])^(-1))
		ωs[t] = Υs[t]*((σs[t])^(-1)*μs[t] + (Ψs[t])^(-1)*ηs[t])
	end

	# t = 0
	Υ_0 = 1 / ((σ_0)^(-1) + (Ψ_0)^(-1))
	ω_0 = Υ_0*((σ_0)^(-1)μ_0 + (Ψ_0)^(-1)η_0)
	
	return ωs, Υs, ω_0, Υ_0
end

function v_pairwise_x(σs_, exp_np::Exp_ϕ_uni, Ψs)
	T = length(σs_)

	# cross-covariance is then computed for all time steps t = 0, ..., T−1
	Υ_ₜ₋ₜ₊₁ = zeros(T)
	
	for t in 1:T-2
		Υ_ₜ₋ₜ₊₁[t+1] = σs_[t+1]*exp_np.A*(1.0 + exp_np.CᵀR⁻¹C + (Ψs[t+1])^(-1) - exp_np.A*σs_[t+1]*exp_np.A)^(-1)
	end

	# t=0, the cross-covariance between the zeroth and first hidden states.
	Υ_ₜ₋ₜ₊₁[1] = σs_[1]*exp_np.A*(1.0 + exp_np.CᵀR⁻¹C + (Ψs[1])^(-1) - exp_np.A*σs_[1]*exp_np.A)^(-1)

	# t=T-1, Ψs[T] = 0 special case
	Υ_ₜ₋ₜ₊₁[T] = σs_[T]*exp_np.A*(1.0 + exp_np.CᵀR⁻¹C - exp_np.A*σs_[T]*exp_np.A)^(-1)
	
	return Υ_ₜ₋ₜ₊₁
end

function vb_e_uni(y::Vector{Float64}, hpp::HPP_uni, exp_np::Exp_ϕ_uni, smooth_out = false)
	T = length(y)

	# forward pass
	μs, σs, σs_ = v_forward(y, exp_np, hpp.μ₀, hpp.σ₀)

	# backward pass 
	ηs, Ψs, η₀, Ψ₀ = v_backward(y, exp_np)

	# marginal (smoothed) means, covs, and pairwise beliefs 
	ωs, Υs, ω_0, Υ_0 = parallel_smoother(μs, σs, ηs, Ψs, η₀, Ψ₀, hpp.μ₀, hpp.σ₀)
	Υ_ₜ₋ₜ₊₁ = v_pairwise_x(σs_, exp_np, Ψs)

	# hidden state sufficient stats 
	W_A = sum(Υs[t-1] + ωs[t-1] * ωs[t-1] for t in 2:T)
	W_A += Υ_0 + ω_0*ω_0

	S_A = sum(Υ_ₜ₋ₜ₊₁[t] + ωs[t-1] * ωs[t] for t in 2:T)
	S_A += Υ_ₜ₋ₜ₊₁[1] + ω_0*ωs[1]
	
	W_C = sum(Υs[t] + ωs[t] * ωs[t] for t in 1:T)
	S_C = sum(ωs[t] * y[t] for t in 1:T)

	if (smooth_out)
		return ωs, Υs
	end
	
	return HSS_uni(W_A, S_A, W_C, S_C), ω_0, Υ_0
end


function vb_dlm(y::Vector{Float64}, hpp::HPP_uni, max_iter=1000)
	T = length(y)
	W_A = 1.0
	S_A = 1.0
	W_C = 1.0
	S_C = 1.0
	
	hss = HSS_uni(W_A, S_A, W_C, S_C)
	exp_np = missing
	
	for i in 1:max_iter
		exp_np = vb_m_uni(y, hss, hpp)
				
		hss, ω_0, Υ_0 = vb_e_uni(y, hpp, exp_np)
	end

	return exp_np
end


function vb_ll_his(y::Vector{Float64}, hpp::HPP_uni, max_iter=1000)
	T = length(y)
	
	hss = HSS_uni(1.0, 1.0, 1.0, 1.0)
	exp_np = missing

	history_A = Vector{Float64}(undef, max_iter - 50)
    history_C = Vector{Float64}(undef, max_iter - 50)
    history_R = Vector{Float64}(undef, max_iter - 50)
	
	for i in 1:max_iter
		exp_np = vb_m_uni(y, hss, hpp)
				
		hss, ω_0, Υ_0 = vb_e_uni(y, hpp, exp_np)

		if(i > 50) # discard the first 10 to see better plots
			history_A[i-50] = exp_np.A
			history_C[i-50] = exp_np.C
       		history_R[i-50] = 1/exp_np.R⁻¹
		end
	end

	return exp_np, history_A, history_C, history_R
end

# Gibbs sampling
function sample_a(xs, q)
	T = length(xs)
	return rand(Normal(sum(xs[1:T-1] .* xs[2:T]) / sum(xs[1:T-1].^2), sqrt(q / sum(xs[1:T-1].^2))))
end

function sample_c(xs, ys, r)
    return rand(Normal(sum(ys .* xs) / sum(xs.^2), sqrt(r / (sum(xs.^2)))))
end

function sample_r(xs, ys, c, α_r, β_r)
	T = length(ys)
    α_post = α_r + T / 2
    β_post = β_r + sum((ys - c * xs).^2) / 2
	λ_r = rand(Gamma(α_post, 1 / β_post))
	return 1/λ_r # inverse precision is variance
end

function sample_x_ffbs(y, A, C, Q, R, μ_0, σ_0)
    T = length(y)
    μs = Vector{Float64}(undef, T)
    σs = Vector{Float64}(undef, T)
    μs[1] = μ_0
    σs[1] = σ_0
	
    for t in 2:T #forward
        μ_pred = A * μs[t-1]
        σ_pred = A * σs[t-1] * A + Q
        K = σ_pred * C * (1/ (C^2 * σ_pred + R))
		
        μs[t] = μ_pred + K * (y[t] - C * μ_pred)
        σs[t] = (1 - K * C) * σ_pred
    end

	x = Vector{Float64}(undef, T)
    x[T] = rand(Normal(μs[T], sqrt(σs[T])))

    for t in (T-1):-1:1 #backward
        μ_cond = μs[t] + σs[t] * A * (1/ (A * σs[t] * A + Q)) * (x[t+1] - A * μs[t])
        σ_cond = σs[t] - σs[t] * A * (1/ (A * σs[t] * A + Q)) * A * σs[t]
        x[t] = rand(Normal(μ_cond, sqrt(σ_cond)))
    end
    return x
end

function gibbs_uni_dlm(y, num_iterations=2000, burn_in=200, thinning=5)
	T = length(y)
	μ_0 = 0.0  # Prior mean for the states
	λ_0 = 1.0  # Prior precision for the states
	α = 0.01  # Shape parameter for Inverse-Gamma prior
	β = 0.01  # Scale parameter for Inverse-Gamma prior
	
	# Initial values for the parameters
	a = rand(Normal(μ_0, λ_0))
	c = rand(Normal(μ_0, λ_0))
	r = rand(InverseGamma(α, β))
	q = 1.0

	n_samples = Int.(num_iterations/thinning)
	# Store the samples
	samples_x = zeros(n_samples, T)
	samples_a = zeros(n_samples)
	samples_c = zeros(n_samples)
	samples_r = zeros(n_samples)
	
	# Gibbs sampler
	for i in 1:num_iterations+burn_in
	    # Update the states
		x = sample_x_ffbs(y, a, c, q, r, μ_0, 1/λ_0)
	
	    # Update the state transition factor
	    a = sample_a(x, q)
	
	    # Update the emission factor
	    c = sample_c(x, y, r)
	
	    # Update the observation noise
		r = sample_r(x, y, c, α, β)
	
	    # Store the samples
		if i > burn_in && mod(i - burn_in, thinning) == 0
			index = div(i - burn_in, thinning)
		    samples_x[index, :] = x
		    samples_a[index] = a
		    samples_c[index] = c
		    samples_r[index] = r
		end
	end
	return samples_x, samples_a, samples_c, samples_r
end

# DLM with R
function sample_q(xs, a, α_q, β_q, x_0)
	T = length(xs)
    α_post = α_q + T / 2
    β_post = β_q + sum((xs[2:T] .- (a .* xs[1:T-1])).^2) /2 
	
	β_post += (xs[1] - a * x_0)^2 /2
	λ_q = rand(Gamma(α_post, 1 / β_post))
	
	return 1/λ_q # inverse precision is variance
end

function gibbs_ll(y, a, c, mcmc=3000, burn_in=1500, thinning=1)
	T = length(y)
	μ_0 = 0.0  # Prior mean for the states
	λ_0 = 1.0  # Prior precision for the states
	
	α = 0.01  # Shape parameter for Inverse-Gamma prior
	β = 0.01  # Scale parameter for Inverse-Gamma prior
	
	# Initial values for the parameters
	r = rand(InverseGamma(α, β))
	q = rand(InverseGamma(α, β))

	n_samples = Int.(mcmc/thinning)
	# Store the samples
	samples_x = zeros(n_samples, T)
	samples_q = zeros(n_samples)
	samples_r = zeros(n_samples)
	
	# Gibbs sampler
	for i in 1:mcmc+burn_in
	    # Update the states
		x = sample_x_ffbs(y, a, c, q, r, μ_0, 1/λ_0)
		
		# Update the system noise
		q = sample_q(x, a, α, β, μ_0)
		
	    # Update the observation noise
		r = sample_r(x, y, c, α, β)
	
	    # Store the samples
		if i > burn_in && mod(i - burn_in, thinning) == 0
			index = div(i - burn_in, thinning)
		    samples_x[index, :] = x
			samples_q[index] = q
		    samples_r[index] = r
		end
	end

	return samples_x, samples_q, samples_r
end

function test_gibbs_ll(rnd, mcmc=10000, burn_in=5000, thin=1)
	Random.seed!(rnd)
	T = 500
	A = 1.0
	C = 1.0
	R = 0.1
	Q = 0.5
	y, x_true = gen_data(A, C, Q, R, 0.0, 1.0, T)
	n_samples = Int.(mcmc/thin)
	println("--- MCMC ---")
	@time s_x, s_q, s_r = gibbs_ll(y, A, C, mcmc, burn_in, thin)
	println("--- n_samples: $n_samples, burn-in: $burn_in, thinning: $thin ---")

	Q_chain = Chains(reshape(s_q, n_samples, 1))
	R_chain = Chains(reshape(s_r, n_samples, 1))

	summary_stats_q = summarystats(Q_chain)
	summary_stats_r = summarystats(R_chain)
	summary_df_q = DataFrame(summary_stats_q)
	summary_df_r = DataFrame(summary_stats_r)
	println("Q summary stats: ", summary_df_q)
	println()
	println("R summary stats: ", summary_df_r)

	x_m = mean(s_x, dims=1)[1,:]
	println("\nend chain x sample error ", error_metrics(x_true, s_x[end,: ]))
	println("average x sample error " , error_metrics(x_true, x_m))
end

begin
	struct HSS_ll
	    w_c::Float64
	    w_a::Float64
	    s_c::Float64
	    s_a::Float64
	end
	
	struct Priors_ll
	    α_r::Float64
	    β_r::Float64
	    α_q::Float64
	    β_q::Float64
	    μ_0::Float64
	    σ_0::Float64
	end

	struct qθ
		α_r_p
		β_r_p
		α_q_p
		β_q_p
	end
end

function vb_m_ll(y, hss::HSS_ll, priors::Priors_ll)
    T = length(y)

    # Update parameters for τ_r
    α_r_p = priors.α_r + T / 2
    β_r_p = priors.β_r + 0.5 * (y' * y - 2 * hss.s_c + hss.w_c)

    # Update parameters for τ_q
    α_q_p = priors.α_q + T / 2
    β_q_p = priors.β_q + 0.5 * (hss.w_a + hss.w_c - 2 * hss.s_a)

    # Compute expectations
    E_τ_r = α_r_p / β_r_p
    E_τ_q = α_q_p / β_q_p

    return E_τ_r, E_τ_q, qθ(α_r_p, β_r_p, α_q_p, β_q_p)
end

function forward_ll(y, a, c, E_τ_r, E_τ_q, priors::Priors_ll)
    T = length(y)
    μ_f = zeros(T)
    σ_f2 = zeros(T)
	fs = zeros(T)
	ss = zeros(T)
	
	a_1 = a * priors.μ_0
	r_1 = a^2 * priors.σ_0 + 1/E_τ_q
	f_1 = c * a_1
	s_1 = c^2 * r_1 + 1/E_τ_r
	fs[1] = f_1
	ss[1] = s_1
	
	μ_f[1] = a_1 + r_1 * c * (1/s_1) * (y[1] - f_1)
    σ_f2[1] = r_1 - r_1^2 * c^2 * (1/s_1)
	
    for t = 2:T
        # Predict step
        μ_pred = a * μ_f[t-1]
        σ_pred2 = a^2 * σ_f2[t-1] + 1/E_τ_q

		f_t = c * μ_pred
		s_t = c^2 * σ_pred2 + 1/E_τ_r

		fs[t] = f_t
		ss[t] = s_t

		μ_f[t] = μ_pred + σ_pred2 * c * (1/s_t) * (y[t] - f_t)
		σ_f2[t] = σ_pred2 - σ_pred2^2 * c^2 * (1/s_t)
        # Update step
        #K_t = σ_pred2 / (σ_pred2 + 1/E_τ_r)
        #μ_f[t] = μ_pred + K_t * (y[t] - μ_pred)
        #σ_f2[t] = (1 - K_t) * σ_pred2
    end

	# dlm pg53. beale p175
	log_z = sum(logpdf(Normal(fs[i], sqrt(ss[i])), y[i]) for i in 1:T)
	
    return μ_f, σ_f2, log_z
end

function backward_ll(μ_f, σ_f2, E_τ_q, priors::Priors_ll)
    T = length(μ_f)
    μ_s = similar(μ_f)
    σ_s2 = similar(σ_f2)
    σ_s2_cross = zeros(T)
    μ_s[T] = μ_f[T]
    σ_s2[T] = σ_f2[T]
    for t = T-1:-1:1
        # Compute the gain J_t
        J_t = σ_f2[t] / (σ_f2[t] + 1/E_τ_q)

        # Update the smoothed mean μ_s and variance σ_s2
        μ_s[t] = μ_f[t] + J_t * (μ_s[t+1] - μ_f[t])
        σ_s2[t] = σ_f2[t] + J_t^2 * (σ_s2[t+1] - σ_f2[t] - 1/E_τ_q)

        # Compute the cross variance σ_s2_cross
        σ_s2_cross[t+1] = J_t * σ_s2[t+1]
    end
	
    #J_0 = σ_f2[1] / (σ_f2[1] + 1/E_τ_q)
	J_0 = 1.0 / (1.0 + 1/E_τ_q)

	μ_s0 = priors.μ_0 + J_0 * (μ_s[1] - priors.μ_0)
	σ_s0 = priors.σ_0 + J_0^2 * (σ_s2[1] - priors.σ_0 - 1/E_τ_q)
	
	σ_s2_cross[1] = J_0 * σ_s2[1]
    return μ_s, σ_s2, μ_s0, σ_s0, σ_s2_cross
end

function vb_e_ll(y, E_τ_r, E_τ_q, priors::Priors_ll)
    # Forward pass (filter)
    μs_f, σs_f2, log_Z = forward_ll(y, 1.0, 1.0, E_τ_r, E_τ_q, priors)

    # Backward pass (smoother)
    μs_s, σs_s2, μs_0, σs_s0, σs_s2_cross = backward_ll(μs_f, σs_f2, E_τ_q, priors)

    # Compute the sufficient statistics
    w_c = sum(σs_s2 .+ μs_s.^2)
    w_a = sum(σs_s2[1:end-1] .+ μs_s[1:end-1].^2)
	w_a += σs_s0 + μs_0^2
	
    s_c = sum(y .* μs_s)
    s_a = sum(σs_s2_cross[1:end-1]) + sum(μs_s[1:end-1] .* μs_s[2:end])
	s_a += μs_0 * μs_s[1]
    # Return the sufficient statistics in a HSS struct
    return HSS_ll(w_c, w_a, s_c, s_a), μs_0, σs_s0, log_Z
end

function vb_ll(y::Vector{Float64}, hpp::Priors_ll, max_iter=100)
	hss = HSS_ll(1.0, 1.0, 1.0, 1.0)
	E_τ_r, E_τ_q  = missing, missing
	
	for i in 1:max_iter
		E_τ_r, E_τ_q, _ = vb_m_ll(y, hss, hpp)
				
		hss, _, _, _ = vb_e_ll(y, E_τ_r, E_τ_q, hpp)
	end

	return 1/E_τ_r, 1/E_τ_q
end

function update_ab(hpp::Priors_ll, qθ)
	exp_r = qθ.α_r_p / qθ.β_r_p
	exp_log_r = digamma(qθ.α_r_p) - log(qθ.β_r_p)
	exp_q = qθ.α_q_p / qθ.β_q_p
	exp_log_q = digamma(qθ.α_q_p) - log(qθ.β_q_p)
	
    d_r, d_q = exp_r, exp_q
    c_r, c_q = exp_log_r, exp_log_q
    
    # Update using fixed point equations
	a_r = hpp.α_r
	a_q = hpp.α_q
	
    for _ in 1:100
        ψ_a = digamma(a_r)
        ψ_a_p = trigamma(a_r)
        a_new = a_r * exp(-(ψ_a - log(a_r) + log(d_r) - c_r) / (a_r * ψ_a_p - 1))
		a_r = a_new
		# check convergence
        if abs(a_new - a_r) < 1e-5
            break
        end
    end
    b_r = a_r/d_r

	for _ in 1:100
		ψ_a_q = digamma(a_q)
        ψ_a_q_p = trigamma(a_q)
        a_new_q= a_q * exp(-(ψ_a_q - log(a_q) + log(d_q) - c_q) / (a_q * ψ_a_q_p - 1))
		a_q = a_new_q

		if abs(a_new_q - a_q) < 1e-5
			break
		end
	end
	b_q = a_q/d_q
	
	return a_r, b_r, a_q, b_q
end

function vb_ll_c(y::Vector{Float64}, hpp::Priors_ll, hp_learn=false, max_iter=500, tol=5e-4)
	hss = HSS_ll(1.0, 1.0, 1.0, 1.0)
	E_τ_r, E_τ_q  = missing, missing
	elbo_prev = -Inf
	el_s = zeros(max_iter)
	for i in 1:max_iter
		E_τ_r, E_τ_q, qθ = vb_m_ll(y, hss, hpp)
		hss, μs_0, σs_s0, log_z = vb_e_ll(y, E_τ_r, E_τ_q, hpp)

		kl_ga = kl_gamma(hpp.α_r, hpp.β_r, qθ.α_r_p, qθ.β_r_p) + kl_gamma(hpp.α_q, hpp.β_q, qθ.α_q_p, qθ.β_q_p)
		elbo = log_z - kl_ga
		el_s[i] = elbo
		
		if abs(elbo - elbo_prev) < tol
			println("Stopped at iteration: $i")
			el_s = el_s[1:i]
            break
		end
		
		if (hp_learn)
			if (i%5 == 0) 
				a_r, b_r, a_q, b_q = update_ab(hpp, qθ)
				hpp = Priors_ll(a_r, b_r, a_q, b_q, μs_0, σs_s0)
			end
		end

        elbo_prev = elbo

		if (i == max_iter)
			println("Warning: VB have not necessarily converged at $max_iter iterations")
		end
	end

	return 1/E_τ_r, 1/E_τ_q, el_s
end

function test_vb_ll(rnd)
	Random.seed!(rnd)
	T = 500
	A = 1.0
	C = 1.0
	R = 0.1
	Q = 0.5
	y, x_true = gen_data(A, C, Q, R, 0.0, 1.0, T)

	println("Ground-truth r: ", R)
	println("Ground-truth q: ", Q)

	model = LocalLevel(y)
	StateSpaceModels.fit!(model) # MLE and uni-variate kalman filter
	print_results(model)

	fm = get_filtered_state(model)
	filter_err = error_metrics(x_true, fm)

	sm = get_smoothed_state(model)
	smooth_err = error_metrics(x_true, sm)

	println("\nPackage Filtered MSE, MAD: ", filter_err)
	println("Package Smoother MSE, MAD: ", smooth_err)
	
	hpp_ll = Priors_ll(0.01, 0.01, 0.01, 0.01, 0.0, 1.0)

	println("\n--- VBEM ---")
	for t in [false, true]
		println("\nHyperparam optimisation: $t")
		@time r, q = vb_ll_c(y, hpp_ll, t)

		println("r: ", r)
		println("q: ", q)
		μs_f, σs_f2 = forward_ll(y, 1.0, 1.0, 1/r, 1/q, hpp_ll)
    	μs_s, _, _ = backward_ll(μs_f, σs_f2, 1/q, hpp_ll)
		println("\n VB latent x error (MSE, MAD) : " , error_metrics(x_true, μs_s))
	end
end

#test_vb_ll()

function main()
	println("Running experiments for local level model:\n")

	seeds = [88, 145, 105, 104, 134]
	#seeds = [103, 133, 100, 143, 111]
	for sd in seeds
		println("\n----- BEGIN Run seed: $sd -----\n")
		test_gibbs_ll(sd, 30000, 5000, 3)
		println()
		test_vb_ll(sd)
		println("----- END Run seed: $sd -----\n")
	ends
end

#main()

function out_txt()
	file_name = "$(splitext(basename(@__FILE__))[1])_$(Dates.format(now(), "yyyymmdd_HHMMSS")).txt"

	# Open a file for writing
	open(file_name, "w") do f
		# Redirect standard output and standard error to the file
		redirect_stdout(f) do
			redirect_stderr(f) do
				# Your script code here
				main()
			end
		end
	end
end

out_txt()

# PLUTO_PROJECT_TOML_CONTENTS = """
# [deps]
# Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
# LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
# MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
# Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
# PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
# Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
# SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
# StateSpaceModels = "99342f36-827c-5390-97c9-d7f9ee765c78"
# StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
# Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"

# [compat]
# Distributions = "~0.25.90"
# MCMCChains = "~6.0.3"
# Plots = "~1.38.11"
# PlutoUI = "~0.7.51"
# SpecialFunctions = "~2.2.0"
# StateSpaceModels = "~0.6.6"
# StatsBase = "~0.33.21"
# Turing = "~0.26.2"
# """