include("kl_optim.jl")
include("turing_ll.jl")
include("test_data.jl")

module LocalLevel

begin
	using Distributions, Random
	using LinearAlgebra
end

export gen_data

function gen_data(A, C, Q, R, m_0, C_0, T)
	x = zeros(T+1)
	y = zeros(T)
	
	x[1] = m_0 
	x[2] = rand(Normal(A*m_0, sqrt(A*C_0*A + Q)))
	y[1] = rand(Normal(C*x[2], sqrt(R)))

	for t in 2:T
		x[t+1] = A * x[t] + sqrt(Q) * randn()
		y[t] = C * x[t+1] + sqrt(R) * randn()
	end
	return y, x
end

end

begin
	using Distributions, Random
	using LinearAlgebra
	using SpecialFunctions
	using MCMCChains
	using DataFrames
	using StatsPlots
	using StateSpaceModels
	using KernelDensity
end

begin
	struct HSS_uni
	    W_A::Float64
	    S_A::Float64
	    W_C::Float64
	    S_C::Float64
	end
end

"""
MCMC
"""
function sample_r(xs, ys, c, α_r, β_r) # Ψ_1
	T = length(ys)
	xs = xs[2:end]
    α_post = α_r + T / 2 #shape
    β_post = β_r + sum((ys - c * xs).^2) / 2 #rate

	# emission precision posterior, Julia Gamma (shpae, 1/rate)
	λ_r = rand(Gamma(α_post, 1 / β_post))
	return 1/λ_r
end

function sample_q(xs, a, α_q, β_q) # Ψ_2
	T = length(xs) # xs length is T+1
    α_post = α_q + (T-1) / 2
    β_post = β_q + sum((xs[2:end] .- (a .* xs[1:end-1])).^2) / 2 
	
	# system precision posterior
	λ_q = rand(Gamma(α_post, 1 / β_post)) 
	return 1/λ_q
end

function test_rq(rnd, T=100)
	R = 1.0
	Q = 10.0
	println("Ground-truth r = $R, q = $Q")

	Random.seed!(rnd)
	y, x_true = LocalLevel.gen_data(1.0, 1.0, Q, R, 0.0, 1.0, T)
	r = sample_r(x_true, y, 1.0, 2.0, 0.001)
	println("Sample r = $r")
	q = sample_q(x_true, 1.0, 2.0, 0.001)
	println("Sample q = $q")
end

# test_rq(123)
# test_rq(123, 1000)
# test_rq(12)
# test_rq(12, 1000)

function forward_filter(y, A, C, R, Q, m_0, c_0)
	T = length(y)
    ms = Vector{Float64}(undef, T+1)
    cs = Vector{Float64}(undef, T+1)
    ms[1] = m_0
    cs[1] = c_0
	a_s = Vector{Float64}(undef, T)
	rs = Vector{Float64}(undef, T)

    for t in 1:T # forward filter
        a_s[t] = a_t = A * ms[t]
        rs[t] = r_t = A * cs[t] * A + Q

		f_t = C * a_t #y pred
		q_t = C * r_t * C + R

        ms[t+1] = a_t + r_t*C*(1/q_t)*(y[t] - f_t)
        cs[t+1] = r_t - r_t*C*(1/q_t)*C*r_t
    end

	return ms, cs, a_s, rs
end

function sample_x_ffbs(y, A, C, R, Q, m_0, c_0)
	T = length(y)
	x = Vector{Float64}(undef, T+1)
	ms, cs, a_s, rs = forward_filter(y, A, C, R, Q, m_0, c_0)
	
	# t = T
    x[end] = rand(Normal(ms[end], sqrt(cs[end])))

    for t in T:-1:1 # backward sample (dlm with R Algorithm 4.1)
        h_t = ms[t] + cs[t] * A * (1 /rs[t]) * (x[t+1] - a_s[t])
        H_t = cs[t] - cs[t] * A * (1/ rs[t]) * A * cs[t]
		x[t] = rand(Normal(h_t, sqrt(H_t)))
    end

    return x
end

function test_ffbs(rnd, T=100)
	R = 1.0
	Q = 1.0
	println("Ground-truth r = $R, q = $Q")

	Random.seed!(rnd)
	y, _ = gen_data(1.0, 1.0, Q, R, 0.0, 1.0, T)

	Xs = zeros(T, 1000)
	for iter in 1:1000
		xs = sample_x_ffbs(y, 1.0, 1.0, R, Q, 0.0, 1e7)
		Xs[:, iter] = xs[2:end]
	end

	p = plot(Xs[1:end, 1:200], label = "", title="FFBS, V=$R, W=$Q")
	display(p)

	x_std = std(Xs, dims=2)[:]
	p = plot(x_std, title="FFBS x std, T=$T", label="")
	display(p)
end

# test_ffbs(111, 20)
# test_ffbs(111, 100)

function gibbs_ll(y, a, c, mcmc=3000, burn_in=100, thinning=1)
	T = length(y)
	m_0, c_0 = 0.0, 1e7  # Prior DLM with R setting
	
	α = 2  # Shape for Gamma prior
	β = 1e-4  # rate for Gamma prior
	
	# Initial values for the parameters, akin to dlm with R
	r = 1.0
	q = 1.0

	n_samples = Int.(mcmc/thinning)
	samples_x = zeros(T, n_samples)
	samples_q = zeros(n_samples)
	samples_r = zeros(n_samples)
	
	for i in 1:mcmc+burn_in
		x = sample_x_ffbs(y, a, c, r, q, m_0, c_0)
		q = sample_q(x, a, α, β)
		r = sample_r(x, y, c, α, β)
		x = x[2:end]

		if i > burn_in && mod(i - burn_in, thinning) == 0
			index = div(i - burn_in, thinning)
		    samples_x[:, index] = x
			samples_q[index] = q
		    samples_r[index] = r
		end
	end
	return samples_x, samples_q, samples_r
end

function test_gibbs_ll(y, x_true=nothing, mcmc=10000, burn_in=5000, thin=1; show_plot=false)
	n_samples = Int.(mcmc/thin)
	println("--- MCMC ---")
	@time s_x, s_q, s_r = gibbs_ll(y, 1.0, 1.0, mcmc, burn_in, thin)
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

	x_m = mean(s_x, dims=2)[:]	
	x_std = std(s_x, dims=2)[:]

	if x_true !== nothing
		println("average x sample error " , error_metrics(x_true[2:end], x_m))
	end

	if show_plot
		# p = plot((s_x[1:end, 1:200]), label = "", title="MCMC Trace Plot x[0:T]")
		# display(p)

		 # Agrees with DLM with R example !
		p = plot(x_std, label = "", title="MCMC x std")
		display(p)

		# p = plot_CI_ll(x_m, x_std, x_true)
		# title!(p, "MCMC latent x inference")
		# display(p)

		# p_q = plot(s_q, title="trace q")
		# display(p_q)

		# p_r = plot(s_r, title="trace r")
		# display(p_r)

		p_r_d = density(R_chain)
		title!(p_r_d, "MCMC R")
		display(p_r_d)

		p_q_d = density(Q_chain)
		title!(p_q_d, "MCMC Q")
		display(p_q_d)

		return x_m, x_std, s_r, s_q
	end

	# empirical truth to compare with VI
	return x_m, x_std, s_r, s_q, R_chain, Q_chain
end

"""
VBEM
"""

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
    #β_r_p = priors.β_r + 0.5 * (y' * y - 2 * hss.s_c + hss.w_c)
	β_r_p = priors.β_r + 0.5 * (y' * y - hss.s_c^2/hss.w_c)
    
	# Update parameters for τ_q
    α_q_p = priors.α_q + (T-1) / 2
    β_q_p = priors.β_q + 0.5 * (hss.w_a + hss.w_c - 2 * hss.s_a)

    # Compute expectations
    E_τ_r = α_r_p / β_r_p
    E_τ_q = α_q_p / β_q_p

	# DEBUG R posterior
    return E_τ_r, E_τ_q, qθ(α_r_p, β_r_p, α_q_p, β_q_p)
end

function forward_ll(y, a, c, E_τ_r, E_τ_q, priors::Priors_ll)
    T = length(y)
	
	#filter state
    μ_f = zeros(T+1)
    σ_f = zeros(T+1)

	#filter preds
	a_s = zeros(T)
	rs = zeros(T)
	
	μ_f[1] = priors.μ_0
	σ_f[1] = priors.σ_0

	log_z = 0.0

    for t in 1:T
        a_s[t] = a_t = a * μ_f[t]
        rs[t] = r_t = a * σ_f[t] * a + 1/E_τ_q # Q

		f_t = c * a_t
		q_t = c * r_t * c + 1/E_τ_r # R
		log_z += logpdf(Normal(f_t, sqrt(q_t)), y[t])

		# m_t, C_t Kalman Filter
		μ_f[t+1] = a_t + r_t * c * (1/q_t) * (y[t] - f_t)
		σ_f[t+1] = r_t - r_t * c * (1/q_t) * c * r_t
    end
	
    return μ_f, σ_f, a_s, rs, log_z
end

function backward_ll(a, μ_f, σ_f, a_s, rs, E_τ_q)
    T = length(μ_f) - 1
    μ_s = similar(μ_f)
    σ_s = similar(σ_f)
    σ_s_cross = zeros(T)

    μ_s[end] = μ_f[end]
    σ_s[end] = σ_f[end]

    for t in T:-1:1 #s_t, S_t, Kalman Smoother
		μ_s[t] = μ_f[t] + σ_f[t] * a * (1/rs[t]) * (μ_s[t+1] - a_s[t])
		σ_s[t] = σ_f[t] - σ_f[t] * a * (1/rs[t]) * (rs[t] - σ_s[t+1]) * (1/rs[t]) * a * σ_f[t]

		J_t = σ_f[t] / (σ_f[t] + 1/E_τ_q)	
		σ_s_cross[t] = J_t * σ_s[t+1]
    end
	
    return μ_s, σ_s, σ_s_cross
end

function vb_e_ll(y, a, c, E_τ_r, E_τ_q, priors::Priors_ll)
    # Forward pass (filter)
    μs_f, σs_f, a_s, rs, log_Z = forward_ll(y, a, c, E_τ_r, E_τ_q, priors)

    # Backward pass (smoother)
    μs_s, σs_s, σs_s_cross = backward_ll(a, μs_f, σs_f, a_s, rs, E_τ_q)

    # Compute the sufficient statistics
    w_c = sum(σs_s[2:end] .+ μs_s[2:end].^2)

    w_a = sum(σs_s[1:end-1] .+ μs_s[1:end-1].^2)
 
	s_c = sum(y .* μs_s[2:end])

    s_a = sum(σs_s_cross) + sum(μs_s[1:end-1] .* μs_s[2:end])

    # Return the sufficient statistics in a HSS struct
    return HSS_ll(w_c, w_a, s_c, s_a), μs_s[1], σs_s[1], log_Z
end

function vb_ll(y::Vector{Float64}, hpp::Priors_ll, max_iter=100)
	hss = HSS_ll(1.0, 1.0, 1.0, 1.0)
	E_τ_r, E_τ_q  = missing, missing
	
	for _ in 1:max_iter
		E_τ_r, E_τ_q, _ = vb_m_ll(y, hss, hpp)
				
		hss, _, _, _ = vb_e_ll(y, 1.0, 1.0, E_τ_r, E_τ_q, hpp)
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
	
    for _ in 1:10
        ψ_a = digamma(a_r)
        ψ_a_p = trigamma(a_r)
        a_new = a_r * exp(-(ψ_a - log(a_r) + log(d_r) - c_r) / (a_r * ψ_a_p - 1))
		a_r = a_new
		# check convergence
        if abs(a_new - a_r) < 5e-5
            break
        end
    end
    b_r = a_r/d_r

	for _ in 1:10
		ψ_a_q = digamma(a_q)
        ψ_a_q_p = trigamma(a_q)
        a_new_q= a_q * exp(-(ψ_a_q - log(a_q) + log(d_q) - c_q) / (a_q * ψ_a_q_p - 1))
		a_q = a_new_q

		if abs(a_new_q - a_q) < 5e-5
			break
		end
	end
	b_q = a_q/d_q
	
	return a_r, b_r, a_q, b_q
end

function vb_ll_c(y::Vector{Float64}, hpp::Priors_ll, hp_learn=false, max_iter=500, tol=1e-4; init="gibbs", debug=false)
	"""
	Random initilisation

	- MLE results r, q, randomness with normal(mle_mean, mle_std)

	- Gibbs Run 1

	- Observation variance 

	"""

	E_τ_r, E_τ_q  = missing, missing
	elbo_prev, el_s = -Inf, zeros(max_iter)
	hss, qθ = missing, missing

	if init == "mle"
		println("--- Using MLE initilaization ---")
		model = StateSpaceModels.LocalLevel(y)
		StateSpaceModels.fit!(model)	
		results = model.results
		mle_ms = results.coef_table.coef
		mle_sterrs = results.coef_table.std_err
		# println(mle_ms, mle_sterrs)

		mle_estimate_R = mle_ms[1] # MLE estimate for R
		mle_estimate_Q = mle_ms[2] # MLE estimate for Q
		se_R = mle_sterrs[1] # Standard error for R
		se_Q = mle_sterrs[2] # Standard error for Q

		# For R
		alpha_R = (mle_estimate_R / se_R)^2
		theta_R = se_R^2 / mle_estimate_R
		r_init = rand(Gamma(alpha_R, theta_R))

		# For Q
		alpha_Q = (mle_estimate_Q / se_Q)^2
		theta_Q = se_Q^2 / mle_estimate_Q
		q_init = rand(Gamma(alpha_Q, theta_Q))

		hss, _, _, _ = vb_e_ll(y, 1.0, 1.0, 1/r_init, 1/q_init, hpp)
		if debug
			println("Q_init: ", q_init)
			println("R_init: ", r_init)
			println("w_c, w_a, s_c, s_a :", hss)
		end
	end

	if init == "gibbs"
		println("--- Using Gibbs 1-step initilaization ---")
		_, sq, sr = gibbs_ll(y, 1.0, 1.0, 1, 0, 1)
		r_init, q_init = sr[1], sq[1]
		hss, _, _, _ = vb_e_ll(y, 1.0, 1.0, 1/r_init, 1/q_init, hpp)

		if debug
			println("Q_init: ", q_init)
			println("R_init: ", r_init)
			println("w_c, w_a, s_c, s_a :", hss)
		end
	end

	if init == "obs"
		y_var = var(y)
		r_init = y_var * (1 + randn() * 0.3) 
		q_init = y_var * (1 + randn() * 0.3)

		hss, _, _, _ = vb_e_ll(y, 1.0, 1.0, 1/r_init, 1/q_init, hpp)

		if debug
			println("Q_init: ", q_init)
			println("R_init: ", r_init)
			println("w_c, w_a, s_c, s_a :", hss)
		end
	end

	if init == "fixed"
		println("--- Fixed Start ---")
		hss = HSS_ll(1.0, 1.0, 1.0, 1.0)
	end

	for i in 1:max_iter
		E_τ_r, E_τ_q, qθ = vb_m_ll(y, hss, hpp)
		hss, μs_0, σs_s0, log_z = vb_e_ll(y, 1.0, 1.0, E_τ_r, E_τ_q, hpp)

		kl_r = kl_gamma(hpp.α_r, hpp.β_r, qθ.α_r_p, qθ.β_r_p)
		kl_q = kl_gamma(hpp.α_q, hpp.β_q, qθ.α_q_p, qθ.β_q_p)
		elbo = log_z - kl_r - kl_q
		el_s[i] = elbo
		
		if debug
			println("\nVB iter $i: ")
			println("\tQ: ", 1/E_τ_q)
			println("\tR: ", 1/E_τ_r)
			println("\tα_r, β_r, α_q, β_q: ", qθ)
			println("\tLog Z: ", log_z)
			println("\tKL q: ", kl_q)
			println("\tKL r: ", kl_r)
			println("\tElbo $i: ", elbo)
		end

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

	return 1/E_τ_r, 1/E_τ_q, el_s, qθ
end

function plot_rq_CI(a_q, b_q, mcmc_s, true_param = nothing)
	mcmc_q = histogram(mcmc_s, bins=200, normalize=:pdf, label="MCMC")
	gamma_dist_q = InverseGamma(a_q, b_q)

	ci_lower = quantile(gamma_dist_q, 0.025)
	ci_upper = quantile(gamma_dist_q, 0.975)

	x = range(extrema(mcmc_s)..., length=200) 
	pdf_values = pdf.(gamma_dist_q, x)
	τ_ = plot!(mcmc_q, x, pdf_values, label="VI", lw=2, xlabel="Var", ylabel="Density")
	
	plot!(τ_, [ci_lower, ci_upper], [0, 0], line=:stem, marker=:circle, color=:red, label="95% CI", lw=2)
	vspan!([ci_lower, ci_upper], fill=:red, alpha=0.1, label=nothing)

	if true_param !== nothing
		vline!(τ_, [true_param], label = "ground_truth", linestyle=:dash, linewidth=2)
	end

	return τ_
end

function plot_CI_ll(μ_s, σ_s2, x_true = nothing, max_T = 30)
	# μ_s :: vector of normal mean
	# σ_s2 :: vector of corresponding variance
	μ_s = μ_s[1:max_T]
	σ_s2 = σ_s2[1:max_T]

	if x_true !== nothing
		x_true = x_true[2:max_T+1]
	end

	# https://en.wikipedia.org/wiki/Standard_error, 95% CI
	lower_bound = μ_s - 1.96 .* sqrt.(σ_s2)
	upper_bound = μ_s + 1.96 .* sqrt.(σ_s2)
	
	T = 1:max_T
	p = plot(T, μ_s, ribbon=(μ_s-lower_bound, upper_bound-μ_s), fillalpha=0.5,
		label="95% CI", linewidth=2)
	
	if x_true !== nothing
		plot!(T, x_true, label = "ground_truth")
	end
	return p
end

function test_vb_ll(y, x_true = nothing, hyperoptim = false; show_plot = false)

	"""
	*** Prior choice and initilisation of VB
	"""
	hpp_ll = Priors_ll(2, 1e-4, 2, 1e-4, 0.0, 1e7)

	println("\n--- VBEM ---")
	println("\nHyperparam optimisation: $hyperoptim")
	@time r, q, els, q_rq = vb_ll_c(y, hpp_ll, hyperoptim, init="gibbs")

	μs_f, σs_f, a_s, rs, _ = forward_ll(y, 1.0, 1.0, 1/r, 1/q, hpp_ll)
	μs_s, σs_s, _ = backward_ll(1.0, μs_f, σs_f, a_s, rs, 1/q)
	x_std = sqrt.(σs_s)

	println("\nVB q(r) mode: ", r)
	println("VB q(q) mode: ", q)
	if x_true !== nothing
		println("\nVB latent x error (MSE, MAD) : " , error_metrics(x_true[2:end], μs_s[2:end]))
	end

	if show_plot
		p = plot(x_std, label = "", title="VI x std")
		display(p)
		
		p = plot(els, label = "elbo", title = "ElBO Progression, Hyperparam optim: $hyperoptim")
		display(p)

		#x_plot = plot_CI_ll(μs_s[2:end], σs_s[2:end], x_true, length(y))
		#title!("Local Level x 95% CI, Hyper-param update: $hyperoptim")
		#display(x_plot)
		#plot_file_name = "$(splitext(basename(@__FILE__))[1])_$(Dates.format(now(), "yyyymmdd_HHMMSS")).svg"
		#savefig(x_plot, joinpath(expanduser("~/Downloads/_graphs"), plot_file_name))
	end

	return μs_s, x_std, q_rq, els
end

function test_MLE(y, x_true = nothing)
	model = StateSpaceModels.LocalLevel(y)
	StateSpaceModels.fit!(model)
	print_results(model)

	# results = model.results
	# println(results.coef_table.coef)
	# println(results.coef_table.std_err)

	if x_true !== nothing
		fm = get_filtered_state(model)
		sm = get_smoothed_state(model)

		filter_err = error_metrics(x_true[2:end], fm)
		smooth_err = error_metrics(x_true[2:end], sm)

		println("\nMLE Filtered MSE, MAD: ", filter_err)
		println("MLE Smoother MSE, MAD: ", smooth_err)
	end
end

function test_Nile_ffbs()
	y = get_Nile()
	y = vec(y)
	y = Float64.(y)
	# R = 15099.8
	# Q = 1468.4
	# T = length(y)
	Random.seed!(10)

	# Xs = zeros(T, 1000)
	# for iter in 1:1000
	# 	# DLM with R set c_0 to 1e7
	# 	xs = sample_x_ffbs(y, 1.0, 1.0, R, Q, 0.0, 1e7)
	# 	Xs[:, iter] = xs[2:end]
	# end

	# FFBS agrees with DLM with R!
	#x_std = std(Xs, dims=2)[:]
	#p = plot(x_std, title="Nile FFBS x std, T=$T", label="")
	#display(p)

	_, _, s_r, s_q = test_gibbs_ll(y, nothing, 1500, 0, 1)
	p_s = scatter(s_r, s_q, label="", xlabel="r", ylabel="q")
	display(p_s)

	kde_rq = kde((s_r, s_q), npoints=(256, 256))
	p = contour(kde_rq.x, kde_rq.y, kde_rq.density, 
				title = "Joint KDE of r and q", xlabel = "r", ylabel = "q")

	#xlims!(p, 0.0, 2.5e4)
	#ylims!(p, 0.0, 4e3)
	display(p)
end

test_Nile_ffbs()

function test_nile()
	y = get_Nile()
	y = vec(y)
	y = Float64.(y)

	test_MLE(y)
	p_elbo = plot()
	p_obs_r = plot()
	p_sys_q = plot()

	for _ in 1:20
		_, _, _, _, r_chain, q_chain = test_gibbs_ll(y, nothing, 30000, 0, 10)
		_, _, q_rq, els = test_vb_ll(y)

		density!(p_obs_r, r_chain)
		density!(p_sys_q, q_chain)

		gamma_dist_q = InverseGamma(q_rq.α_q_p, q_rq.β_q_p)
		x = range(0, 2000, length=500) 
		pdf_values = pdf.(gamma_dist_q, x)
		plot!(p_sys_q, x, pdf_values, label="", lw=1, linestyle=:dash, xlabel="Sys q", ylabel="Density", title="q")
		gamma_dist_r = InverseGamma(q_rq.α_r_p, q_rq.β_r_p)
		x = range(10000, 30000, length=500) 
		pdf_values = pdf.(gamma_dist_r, x)
		plot!(p_obs_r, x, pdf_values, label="", lw=1, linestyle=:dash, xlabel="Obs r", ylabel="Density", title="r")

		plot!(p_elbo, els, label="", ylabel="ELBO", xlabel="Iterations")
	end

	display(p_elbo)
	xlims!(p_sys_q, 0, 2000)
	display(p_sys_q)
	xlims!(p_sys_q, 10000, 30000)
	display(p_obs_r)
end

#test_nile()

function compare_mcmc_vi(mcmc::Vector{T}, vi::Vector{T}) where T
    # Ensure all vectors have the same length
    @assert length(mcmc) == length(vi) "All vectors must have the same length"
    
	p_mcmc = scatter(mcmc, vi, label="MCMC", color=:red, alpha=0.5)
	p_vi = scatter!(p_mcmc, mcmc, vi, label="VI", ylabel = "VI", color=:green, alpha=0.5)

	# Determine the range for the y=x line
	min_val = min(minimum(mcmc), minimum(vi))
	max_val = max(maximum(mcmc), maximum(vi))

	# Plot the y=x line
	plot!(p_vi, [min_val, max_val], [min_val, max_val], linestyle=:dash, label = "", color=:blue, linewidth=2)

	return p_vi
end

function main(max_T)
	println("Running experiments for local level model:\n")
	println("T = $max_T")
	R = 50.0
	Q = 100.0
	println("Ground-truth r = $R, q = $Q")

	#seeds = [88, 145, 105, 104, 134]
	seeds = [103, 133, 100, 143, 111]
	for sd in seeds
		println("\n----- BEGIN Run seed: $sd -----\n")
		Random.seed!(sd)
		y, x_true = LocalLevel.gen_data(1.0, 1.0, Q, R, 0.0, 1.0, max_T)
		test_gibbs_ll(y, x_true, 10000, 5000, 1)
		test_vb_ll(y, x_true)
	end
end

function main_graph(sd, max_T=100, sampler="gibbs")
	println("Running experiments for local level model (with graphs):\n")
	println("T = $max_T")
	R = 10000.0
	Q = 1000.0
	println("Ground-truth r = $R, q = $Q")
	Random.seed!(sd)
	y, x_true = LocalLevel.gen_data(1.0, 1.0, Q, R, 0.0, 1.0, max_T)

	vb_x_m, vb_x_std, q_rq, _ = test_vb_ll(y, x_true, show_plot = true)

	"""
	Choice of Gibbs, NUTS, HMC
	"""
	mcmc_x_m, mcmc_x_std, rs, qs = missing, missing, missing, missing
	
	if sampler == "gibbs" #default
		mcmc_x_m, mcmc_x_std, rs, qs = test_gibbs_ll(y, x_true, 10000, 5000, 1, show_plot=true)
	end

	if sampler == "hmc" #turing
		mcmc_x_m, mcmc_x_std, rs, qs = test_hmc(y)
	end

	if sampler == "nuts" #turing
		mcmc_x_m, mcmc_x_std, rs, qs = test_nuts(y)
	end

	kde_result = kde((rs, qs))
	p = contour(kde_result.x, kde_result.y, kde_result.density, 
				title = "Joint KDE of r and q", xlabel = "r", ylabel = "q")

	display(p)

	plot_r, plot_q = plot_rq_CI(q_rq.α_r_p, q_rq.β_r_p, rs, R), plot_rq_CI(q_rq.α_q_p, q_rq.β_q_p, qs, Q)
	display(plot_r)
	display(plot_q)

	p = compare_mcmc_vi(mcmc_x_m, vb_x_m[2:end])
	title!(p, "Latent X inference mean")
	xlabel!(p, "MCMC($sampler)")
	display(p)

	p2 = compare_mcmc_vi(mcmc_x_std, vb_x_std[2:end])
	title!(p2, "Latent X std")
	xlabel!(p2, "MCMC($sampler)")
	display(p2)
end

#main_graph(10, 500, "gibbs")

function out_txt(n)
	file_name = "$(splitext(basename(@__FILE__))[1])_$(Dates.format(now(), "yyyymmdd_HHMMSS")).txt"

	open(file_name, "w") do f
		redirect_stdout(f) do
			redirect_stderr(f) do
				main(n)
			end
		end
	end
end

#out_txt(500)

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