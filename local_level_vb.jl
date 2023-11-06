include("kl_optim.jl")
include("turing_ll.jl")
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
function sample_r(xs, ys, c, α_r, β_r)
	T = length(ys)
    α_post = α_r + T / 2
    β_post = β_r + sum((ys - c * xs).^2) / 2

	# emission precision posterior
	λ_r = rand(Gamma(α_post, 1 / β_post))
	return 1/λ_r
end

function sample_q(xs, a, α_q, β_q)
	T = length(xs)
    α_post = α_q + T / 2
    β_post = β_q + sum((xs[2:end] .- (a .* xs[1:end-1])).^2) / 2 
	
	# system precision posterior
	λ_q = rand(Gamma(α_post, 1 / β_post)) 
	
	return 1/λ_q
end

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

		f_t = C * a_t
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

function gibbs_ll(y, a, c, mcmc=3000, burn_in=100, thinning=1)
	T = length(y)
	m_0 = 0.0  # Prior mean for the states
	c_0 = 1.0  # Prior precision for the states
	
	α = 0.1  # Shape for Inverse-Gamma prior
	β = 0.1  # Scale for Inverse-Gamma prior
	
	# Initial values for the parameters
	r = rand(InverseGamma(α, β))
	q = rand(InverseGamma(α, β))

	n_samples = Int.(mcmc/thinning)
	samples_x = zeros(n_samples, T)
	samples_q = zeros(n_samples)
	samples_r = zeros(n_samples)
	
	for i in 1:mcmc+burn_in
		x = sample_x_ffbs(y, a, c, r, q, m_0, c_0)
		x = x[2:end]
		q = sample_q(x, a, α, β)
		r = sample_r(x, y, c, α, β)
	
		if i > burn_in && mod(i - burn_in, thinning) == 0
			index = div(i - burn_in, thinning)
		    samples_x[index, :] = x
			samples_q[index] = q
		    samples_r[index] = r
		end
	end
	return samples_x, samples_q, samples_r
end

function test_gibbs_ll(y, x_true, mcmc=3000, burn_in=1000, thin=1, show_plot=false)
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

	x_m = mean(s_x, dims=1)[1, :]
	println("average x sample error " , error_metrics(x_true, x_m))
	x_std = std(s_x, dims=1)[1, :]
	x_var = var(s_x, dims=1)[1, :]

	# DEBUG here
	p = plot(s_x'[1:end, 1:200], label = "", title="MCMC Trace Plot x[0:T]")
	display(p)

	if show_plot
		p_r = density(R_chain)
		title!(p_r, "MCMC R")
		display(p_r)

		p_q = density(Q_chain)
		title!(p_q, "MCMC Q")
		display(p_q)

		p = plot_CI_ll(x_m, x_std, x_true)
		title!(p, "MCMC latent x inference")
		display(p)
	end

	# empirical truth to compare with VI
	return x_m, x_var, s_r, s_q
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
	
    return μ_f, σ_f2, fs, ss, log_z
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
    μs_f, σs_f2, _, _ , log_Z = forward_ll(y, 1.0, 1.0, E_τ_r, E_τ_q, priors)

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

function vb_ll_c(y::Vector{Float64}, hpp::Priors_ll, hp_learn=false, max_iter=500, tol=5e-4)
	hss = HSS_ll(1.0, 1.0, 1.0, 1.0)
	E_τ_r, E_τ_q  = missing, missing
	elbo_prev = -Inf
	el_s = zeros(max_iter)
	qθ = missing
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

	return 1/E_τ_r, 1/E_τ_q, el_s, qθ
end

function plot_rq_CI(q::qθ, p_r, p_q)
	mcmc_q = histogram(1 ./ p_q, bins=200, normalize=:pdf, label="MCMC τ_q")
	gamma_dist_q = Gamma(q.α_q_p, 1/q.β_q_p) 
	ci_lower = quantile(gamma_dist_q, 0.025)
	ci_upper = quantile(gamma_dist_q, 0.975)

	x = range(extrema(1 ./p_q)..., length=100) 
	pdf_values = pdf.(gamma_dist_q, x)
	τ_q = plot!(mcmc_q, x, pdf_values, label="VI τ_q", lw=2, xlabel="Precision", ylabel="Density")
	
	plot!(τ_q, [ci_lower, ci_upper], [0, 0], line=:stem, marker=:circle, color=:red, label="95% CI", lw=2)
	vspan!([ci_lower, ci_upper], fill=:red, alpha=0.2, label=nothing)

	mcmc_r = histogram(1 ./ p_r, bins=200, normalize=:pdf, label="MCMC τ_r")
	gamma_dist_r = Gamma(q.α_r_p, 1/q.β_r_p) 
	ci_lower = quantile(gamma_dist_r, 0.025)
	ci_upper = quantile(gamma_dist_r, 0.975)

	x = range(extrema(1 ./ p_r)..., length=100)
	pdf_values = pdf.(gamma_dist_r, x)
	τ_r = plot!(mcmc_r, x, pdf_values, label="VI τ_r", lw=2, xlabel="Precision", ylabel="Density")

	plot!(τ_r, [ci_lower, ci_upper], [0, 0], line=:stem, marker=:circle, color=:red, label="95% CI", lw=2)
	vspan!([ci_lower, ci_upper], fill=:red, alpha=0.2, label=nothing)

	return τ_q, τ_r
end

function plot_CI_ll(μ_s, σ_s2, x_true, max_T = 20)
	# μ_s :: vector of normal mean
	# σ_s2 :: vector of corresponding variance
	μ_s = μ_s[1:max_T]
	σ_s2 = σ_s2[1:max_T]
	x_true = x_true[1:max_T]

	# https://en.wikipedia.org/wiki/Standard_error, 95% CI
	lower_bound = μ_s - 1.96 .* sqrt.(σ_s2)
	upper_bound = μ_s + 1.96 .* sqrt.(σ_s2)
	
	# Create time variable for x-axis
	T = 1:max_T
	p = plot()

	# Plot the smoothed means and the 95% CI
	plot!(T, μ_s, ribbon=(μ_s-lower_bound, upper_bound-μ_s), fillalpha=0.5,
		label="95% CI", linewidth=2)
	
	plot!(T, x_true, label = "ground_truth")
	return p
end

function test_vb_ll(y, x_true, hyperoptim = false, show_plot = false)
	model = LocalLevel(y)
	StateSpaceModels.fit!(model) # MLE and uni-variate kalman filter
	print_results(model)

	fm = get_filtered_state(model)
	filter_err = error_metrics(x_true, fm)

	sm = get_smoothed_state(model)
	smooth_err = error_metrics(x_true, sm)

	println("\nMLE Filtered MSE, MAD: ", filter_err)
	println("MLE Smoother MSE, MAD: ", smooth_err)
	
	hpp_ll = Priors_ll(0.1, 0.1, 0.1, 0.1, 0.0, 1.0)
	println("\n--- VBEM ---")
	println("\nHyperparam optimisation: $hyperoptim")
	@time r, q, els, q_rq = vb_ll_c(y, hpp_ll, hyperoptim)

	if show_plot
		p = plot(els, label = "elbo", title = "ElBO Progression, Hyperparam optim: $hyperoptim")
		display(p)
	end

	μs_f, σs_f2, _, _, _ = forward_ll(y, 1.0, 1.0, 1/r, 1/q, hpp_ll)
	μs_s, σs_s, _ = backward_ll(μs_f, σs_f2, 1/q, hpp_ll)
	x_var = σs_s

	println("\nVB q(r) mode: ", r)
	println("VB q(q) mode: ", q)
	println("\nVB latent x error (MSE, MAD) : " , error_metrics(x_true, μs_s))

	# p = plot(σs_s, label="x var")
	# display(p)

	if show_plot
		x_plot = plot_CI_ll(μs_s, σs_s, x_true)
		title!("Local Level x 95% CI, Hyper-param update: $hyperoptim")
		display(x_plot)
		#plot_file_name = "$(splitext(basename(@__FILE__))[1])_$(Dates.format(now(), "yyyymmdd_HHMMSS")).svg"
		#savefig(x_plot, joinpath(expanduser("~/Downloads/_graphs"), plot_file_name))
	end

	# to compare with MCMC
	return μs_s, x_var, q_rq
end

"""
On-going FFBS debug
"""
function test()
	println("Running experiments for local level model (with graphs):\n")
	R = 1.0
	Q = 1.0
	println("Ground-truth r = $R, q = $Q")

	test_size = 50
	Random.seed!(111)
	y, x_true = gen_data(1.0, 1.0, Q, R, 0.0, 1.0, test_size)
	println("test q: ", sample_q(x_true, 1.0, 0.1, 0.1))
	println("test r: ", sample_r(x_true, y, 1.0, 0.1, 0.1))

	Xs = zeros(test_size, 2000)
	for iter in 1:2000
		xs = sample_x_ffbs(y, 1.0, 1.0, R, Q, 0.0, 1.0)
		Xs[:, iter] = xs[2:end]
	end

	p = plot(Xs[1:end, 1:200], label = "", title="FFBS, V=$R, W=$Q")
	display(p)

	test_gibbs_ll(y, x_true)
end

#test()

function compare_mcmc_vi(mcmc::Vector{T}, vi::Vector{T}) where T
    # Ensure all vectors have the same length
    @assert length(mcmc) == length(vi) "All vectors must have the same length"
    
	p_mcmc = scatter(mcmc, vi, label="MCMC", xlabel = "MCMC", color=:red, alpha=0.6)
	p_vi = scatter!(p_mcmc, mcmc, vi, label="VI", ylabel = "VI", color=:green, alpha=0.3)

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
	R = 0.1
	Q = 1.0
	println("Ground-truth r = $R, q = $Q")

	seeds = [88, 145, 105, 104, 134]
	#seeds = [103, 133, 100, 143, 111]
	for sd in seeds
		println("\n----- BEGIN Run seed: $sd -----\n")
		Random.seed!(sd)
		y, x_true = gen_data(1.0, 1.0, Q, R, 0.0, 1.0, max_T)
		test_gibbs_ll(y, x_true, 10000, 5000, 1)
		test_vb_ll(y, x_true)
	end
end

function main_graph(sd, max_T=100)
	println("Running experiments for local level model (with graphs):\n")
	println("T = $max_T")
	R = 1.0
	Q = 1.0
	println("Ground-truth r = $R, q = $Q")
	Random.seed!(sd)
	y, x_true = gen_data(1.0, 1.0, Q, R, 0.0, 1.0, max_T)

	"""
	Choice of Gibbs, NUTS, HMC
	"""
	# mcmc_x_m, mcmc_x_var, rs, qs = test_gibbs_ll(y, x_true, 3000, 1000, 1)
	# mcmc_x_m, mcmc_x_var, rs, qs = test_nuts(y)
	mcmc_x_m, mcmc_x_var, rs, qs = test_hmc(y)

	vb_x_m, vb_x_var, q_rq = test_vb_ll(y, x_true)

	plot_r, plot_q = plot_rq_CI(q_rq, rs, qs)
	display(plot_r)
	display(plot_q)

	p = compare_mcmc_vi(mcmc_x_m, vb_x_m)
	title!(p, "Latent X inference mean")
	display(p)

	p2 = compare_mcmc_vi(mcmc_x_var, vb_x_var)
	title!(p2, "Latent X Var")
	display(p2)

	p4 = qqplot(mcmc_x_m, vb_x_m, qqline = :R, title="QQ_plot Mean", xlabel="MCMC(HMC)", ylabel="VI")
	display(p4)

	p3 = qqplot(mcmc_x_var, vb_x_var, qqline = :R, title="QQ_plot Var", xlabel="MCMC(HMC)", ylabel="VI")
	xlims!(p3, 0.40, 0.50)
	ylims!(p3, 0.40, 0.50)
	display(p3)

	p5 = qqplot(vb_x_var, mcmc_x_var, qqline = :R, title="QQ_plot Var", xlabel="VI", ylabel="MCMC(HMC)")
	xlims!(p5, 0.40, 0.50)
	ylims!(p5, 0.40, 0.50)
	display(p5)
end

main_graph(111, 100)

function out_txt(n)
	file_name = "$(splitext(basename(@__FILE__))[1])_$(Dates.format(now(), "yyyymmdd_HHMMSS")).txt"

	# Open a file for writing
	open(file_name, "w") do f
		# Redirect standard output and standard error to the file
		redirect_stdout(f) do
			redirect_stderr(f) do
				# Your script code here
				main(n)
			end
		end
	end
end

#out_txt(200)

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