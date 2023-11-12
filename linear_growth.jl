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
	using StateSpaceModels
end

function gen_data(A, C, Q, R, μ_0, Σ_0, T)
	K, _ = size(A) # K = 2
	D, _ = size(C)
	x = zeros(K, T+1)
	y = zeros(D, T) # D = 1

	x[:, 1] = zeros(K)
	x[:, 2] = rand(MvNormal(A*μ_0, A'*Σ_0*A + Q))
	y[:, 1] = C * x[:, 1] + rand(MvNormal(zeros(D), R))

	for t in 2:T
		x[:, t+1] = A * x[:, t] + rand(MvNormal(zeros(K), Q))
		y[:, t] = C * x[:, t+1] + rand(MvNormal(zeros(D), sqrt.(R))) 
	end

	return y, x
end

"""
VBEM
"""
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

	"""
	Q inference needs potential check
	"""
	Exp_Q⁻¹= diagm(mean.(q_𝛐))
	return Exp_R⁻¹, Exp_Q⁻¹, Q_Gamma(a_, b_s, α_, β_s)
end

function forward_(y::Array{Float64, 2}, A::Array{Float64, 2}, C::Array{Float64, 2}, E_R, E_Q::Array{Float64, 2}, prior::HPP_D)
    _, T = size(y)
    K = size(A, 1)
    
    μ_f = zeros(K, T+1)
    Σ_f = zeros(K, K, T+1)

	A_s = zeros(K, T)
	Rs = zeros(K, K, T)
	
    μ_f[:, 1] = prior.μ_0
    Σ_f[:, :, 1] = prior.Σ_0
    
	log_z = 0.0
    
    for t in 1:T # Forward pass (Kalman Filter)
        A_s[:, t] = a_t = A * μ_f[:, t]
        Rs[:, :, t] = R_t = A * Σ_f[:, :, t] * A' + E_Q

		f_t = C * a_t
		Q_t = C * R_t * C' + E_R
		log_z += logpdf(MvNormal(f_t, Symmetric(Q_t)), y[:, t])

		μ_f[:, t+1] = a_t + R_t * C' * inv(Q_t) * (y[:, t] - f_t)
		Σ_f[:, :, t+1] = R_t - R_t * C' * inv(Q_t) * C * R_t
    end
	    
	return μ_f, Σ_f, A_s, Rs, log_z
end

function backward_(A::Array{Float64, 2}, μ_f::Array{Float64, 2}, Σ_f::Array{Float64, 3}, A_s::Array{Float64, 2}, Rs::Array{Float64, 3})
    K, T = size(A_s)
    μ_s = zeros(K, T+1)
    Σ_s = zeros(K, K, T+1)
    Σ_s_cross = zeros(K, K, T)
    
    μ_s[:, end] = μ_f[:, end]
    Σ_s[:, :, end] = Σ_f[:, :, end]
    
    for t in T:-1:1 # Backward pass (Kalman Smoother)
		μ_s[:, t] = μ_f[:, t] + Σ_f[:, :, t] * A' * inv(Rs[:, :, t]) * (μ_s[:, t+1] - A_s[:, t])
		Σ_s[:, :, t] =  Σ_f[:, :, t] - Σ_f[:, :, t] * A' * inv(Rs[:, :, t]) * (Rs[:, :, t] - Σ_s[:, :, t+1]) * inv(Rs[:, :, t]) * A * Σ_f[:, :, t]
		Σ_t_ = inv(inv(Σ_f[:, :, t]) + A'*A)
		Σ_s_cross[:, :, t] = Σ_t_ * A' * Σ_s[:, :, t]
    end
	
    return μ_s, Σ_s, Σ_s_cross
end

function vb_e_step(y::Array{Float64, 2}, A::Array{Float64, 2}, C::Array{Float64, 2}, E_R, E_Q::Array{Float64, 2}, prior::HPP_D)
    # Forward-Backward
	μ_f, Σ_f, A_s, Rs, log_Z = forward_(y, A, C, E_R, E_Q, prior)
    μ_s, Σ_s, Σ_s_cross = backward_(A, μ_f, Σ_f, A_s, Rs)

    # Compute the hidden state sufficient statistics
    W_C = sum(Σ_s[:, :, 2:end], dims=3)[:, :, 1] + μ_s[:, 2:end] * μ_s[:, 2:end]'
    W_A = sum(Σ_s[:, :, 1:end-1], dims=3)[:, :, 1] + μ_s[:, 1:end-1] * μ_s[:, 1:end-1]'
    S_C = μ_s[:, 2:end] * y'
    S_A = sum(Σ_s_cross, dims=3)[:, :, 1] + μ_s[:, 1:end-1] * μ_s[:, 2:end]'

	# Return the hidden state sufficient statistics
    return HSS(W_C, W_A, S_C, S_A), μ_s[:, 1], Σ_s[:, :, 1], log_Z
end

function vbem_lg(y::Array{Float64, 2}, A::Array{Float64, 2}, C::Array{Float64, 2}, prior::HPP_D, max_iter=100)
	hss = HSS(ones(size(A)), ones(size(A)), ones(size(C')), ones(size(A)))
	E_R_inv, E_Q_inv = missing, missing

	for _ in 1:max_iter
		E_R_inv, E_Q_inv, _ = vb_m_step(y, hss, prior, A, C)		
		hss, _, _, _ = vb_e_step(y, A, C, inv(E_R_inv), inv(E_Q_inv), prior)
	end
	
	return E_R_inv, E_Q_inv
end

function vbem_lg_c(y, A::Array{Float64, 2}, C::Array{Float64, 2}, prior::HPP_D, hp_learn=false, max_iter=500, tol=5e-4)
	D, _ = size(y)
	K = size(A, 1)
	hss = HSS(ones(size(A)), ones(size(A)), ones(size(C')), ones(size(A)))
	E_R_inv, E_Q_inv = missing, missing
	elbo_prev = -Inf
	el_s = zeros(max_iter)
	Q_gam = missing

	for i in 1:max_iter
		E_R_inv, E_Q_inv, Q_gam = vb_m_step(y, hss, prior, A, C)
		hss, μ_s0, Σ_s0, log_Z = vb_e_step(y, A, C, inv(E_R_inv), inv(E_Q_inv), prior)

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
	
	return inv(E_R_inv), inv(E_Q_inv), el_s, Q_gam
end

"""
MCMC
"""
function sample_R(Xs, Ys, C, a_ρ, b_ρ)
    _, T = size(Ys)
	Xs = Xs[:, 2:end]

	Y = Ys[1, :]
	a_post = a_ρ + T / 2 #shape
	b_post = b_ρ + 0.5 * sum((Y' - C[1, :]' * Xs).^2) #rate
	ρ_sampled = rand(Gamma(a_post, 1 / b_post)) #shape, 1/rate

	# inverse precision is variance
    return diagm([1 / ρ_sampled])
end

function sample_Q(Xs, A, α_q, β_q)
    K, T = size(Xs)
    q_sampled = zeros(K)
    for i in 1:K
        X_diff = Xs[i, 2:end] - (A * Xs[:, 1:end-1])[i, :]
        α_post = α_q + (T-1) / 2 
        β_post = β_q + 0.5 * sum(X_diff.^2)
        
        q_sampled[i] = rand(Gamma(α_post, 1 / β_post)) #Julia gamma (shape, 1/rate)
    end
    return diagm(1 ./ q_sampled) # inverse precision is variance
end

function test_Gibbs_RQ(rnd, T = 100)
	A_lg = [1.0 1.0; 0.0 1.0]
    C_lg = [1.0 0.0]
	Q_lg = Diagonal([10.0, 15.0])
	R_lg = [30.0]
	μ_0 = [0.0, 0.0]
	Σ_0 = Diagonal([100.0, 1000.0])
	println("R True:")
	show(stdout, "text/plain", R_lg)
	println()
	println("Q True:")
	show(stdout, "text/plain", Q_lg)
	println()

	Random.seed!(rnd)
	y, x_true = gen_data(A_lg, C_lg, Q_lg, R_lg, μ_0, Σ_0, T)
	prior = Q_Gamma(2, 0.001, 2, 0.001)

	R = sample_R(x_true, y, C_lg, prior.a, prior.b)
	println("\nR Sample ", R)

	Q = sample_Q(x_true, A_lg, prior.α, prior.β)
	println("\nQ Sample ", Q)
end

# R, Q sample methods tested !
# test_Gibbs_RQ(123, 100)
# test_Gibbs_RQ(123, 1000)
# test_Gibbs_RQ(10, 100)
# test_Gibbs_RQ(10, 1000)

function forward_filter(Ys, A, C, R, Q, m_0, C_0)
	"""
	A : State transition (2 X 2)
	C : Emission (2 X 1)
	R : Observation noise (1 X 1)
	Q : System noise (diagonal) (2 X 2)
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

		# filter 
		ms[:, t+1] = a_t + R_t * C' * inv(Q_t) * (Ys[:, t] - f_t)
		Cs[:, :, t+1]= R_t - R_t * C' * inv(Q_t) * C * R_t
	end
	return ms, Cs, a_s, Rs
end

# On-going FFBS Debug
function ffbs_x(Ys, A, C, R, Q, m_0, C_0)
	_, T = size(Ys)
    K, _ = size(A)
	X = zeros(K, T+1)

	ms, Cs, a_s, Rs = forward_filter(Ys, A, C, R, Q, m_0, C_0)
	
	try
		X[:, end] = rand(MvNormal(ms[:, end], Cs[:, :, end]))
	catch PosDefException
		println("PosDefException at t=$T")
		U, Σ, V = svd(Cs[:, :, end])
		H_T = U * diagm(Σ) * V'
 		X[:, end] = rand(MvNormal(ms[:, end], Symmetric(H_T)))
	end

	for t in T:-1:1 # backward sampling
		C_t = Cs[:, :, t]
		# U_c, Σ_c, V_c = svd(Cs[:, :, t])
		# C_t = U_c * diagm(Σ_c) * V_c'

		h_t = ms[:, t] + C_t * A' * inv(Rs[:, :, t])*(X[:, t+1] - a_s[:, t])
		H_t = C_t - C_t * A' * inv(Rs[:, :, t]) * A * C_t
		
		try
			U, Σ, V = svd(H_t)
			H_t = U * diagm(Σ) * V'
			X[:, t] = rand(MvNormal(h_t, Symmetric(H_t)))
		catch PosDefException
			println("PosDefException at t=$t")
			# println("\tH_t ", H_t)
			# U, Σ, V = svd(H_t)
			# H_t = U * diagm(Σ) * V'
			# println("\tH_t (svd)", H_t)
		end
	end

	return X
end

"""
Check FFBS in Linear Growth [ Compare with DLM with R ]
"""

function gibbs_lg(y, A, C, prior::HPP_D, mcmc=10000, burn_in=5000, thinning=1, debug=false)
	P, T = size(y)
	K = size(A, 2)
	
	m_0 = prior.μ_0
	C_0 = prior.Σ_0
	# C_0 = Diagonal(ones(K))
	a, b, α, β = prior.a, prior.b, prior.α, prior.β

	# akin to DLM with R, initilaize all diagonal elements to 1.0
    R = Diagonal(ones(P))
    Q = Diagonal(ones(K))

	n_samples = Int.(mcmc/thinning)
	samples_X = zeros(n_samples, K, T)
	samples_Q = zeros(n_samples, K, K)
	samples_R = zeros(n_samples, P, P)
	
	# Gibbs sampler
	for i in 1:mcmc+burn_in
		if debug
			println("Current iteration: $i")
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

function test_gibbs(y, x_true=nothing, mcmc=10000, burn_in=5000, thin=1, show_plot=false)
	A_lg = [1.0 1.0; 0.0 1.0]
    C_lg = [1.0 0.0]
	K = size(A_lg, 1)
	prior = HPP_D(2, 0.01, 2, 0.01, zeros(K), Matrix{Float64}(I * 1e3, K, K))
	n_samples = Int.(mcmc/thin)

	println("--- MCMC ---")
	@time Xs_samples, Qs_samples, Rs_samples = gibbs_lg(y, A_lg, C_lg, prior, mcmc, burn_in, thin)
	println("\n--- n_samples: $n_samples, burn-in: $burn_in, thinning: $thin ---\n")
	Q_chain = Chains(reshape(Qs_samples, n_samples, 4))
	R_chain = Chains(reshape(Rs_samples, n_samples, 1))

	summary_stats_q = summarystats(Q_chain)
	summary_stats_r = summarystats(R_chain)
	summary_df_q = DataFrame(summary_stats_q)
	summary_df_r = DataFrame(summary_stats_r)
	summary_df_q = summary_df_q[[1, 4], :]
	println("Q summary stats: ", summary_df_q)
	println()
	println("R summary stats: ", summary_df_r)

	xs_m = mean(Xs_samples, dims=1)[1, :, :]
	xs_std = std(Xs_samples, dims=1)[1, :, :]
	if x_true !== nothing
		println("\nMSE, MAD of MCMC X mean: ", error_metrics(x_true[:, 2:end], xs_m))
	end

	if show_plot
		R_chain = Chains(reshape(1 ./ Rs_samples, n_samples, 1))
		p_r = density(R_chain[:, 1, :], label="MCMC Λ_R")
		title!(p_r, "MCMC Λ_R")
		display(p_r)

		p_r_his = histogram(R_chain[:, 1, :], bins=200, normalize=:pdf, label="MCMC Λ_R")
		display(p_r_his)

		Q_chain = Chains(reshape(1 ./ Qs_samples, n_samples, 4))
		p1 = density(Q_chain[:, 1, :], label="MCMC Λ_Q[1, 1]")
		display(p1)

		p1_his = histogram(Q_chain[:, 1, :], bins=200, normalize=:pdf, label="MCMC Λ_Q[1, 1]")
		display(p1_his)

		p4 = density(Q_chain[:, end, :], label="MCMC Λ_Q[2, 2]")
		display(p4)

		p4_his = histogram(Q_chain[:, end, :], bins=200, normalize=:pdf, label="MCMC Λ_Q[2, 2]")
		display(p4_his)

		ps = plot_x_itvl(xs_m, xs_std, x_true, 20)
		for i in 1:K
			p = ps[i]
			title!(p, "MCMC latent x inference")
			#display(p)
			sleep(1)
		end

		return xs_m, xs_std, p_r, p1, p4
	end
	return xs_m, xs_std
end

"""
On-going Gibbs debug, PosDefException for back sample at t=2, need to use SVD !
"""
# A_lg = [1.0 1.0; 0.0 1.0]
# C_lg = [1.0 0.0]
# Q = Diagonal([1.0, 1.0])
# R = [10.0]
# K = size(A_lg, 1)
# Random.seed!(111)
# y, x_true = gen_data(A_lg, C_lg, Q, R, zeros(K), Diagonal(ones(K)), 500)
# test_gibbs(y, x_true, 5000, 1000, 1)

function test_vb(y, x_true, hyperoptim=false, show_plot=false)
	T = size(y, 2)
	A_lg = [1.0 1.0; 0.0 1.0]
    C_lg = [1.0 0.0]
	K = size(A_lg, 1)
	prior = HPP_D(2, 0.01, 2, 0.01, zeros(K), Matrix{Float64}(I, K, K))
	println("--- VBEM ---")

	println("\nHyper-param optimisation: $hyperoptim")
	@time R, Q, elbos, Q_gam = vbem_lg_c(y, A_lg, C_lg, prior, hyperoptim)

	μs_f, σs_f2, A_s, Rs, _ = forward_(y, A_lg, C_lg, R, Q, prior)
	μs_s, Σ_s, _ = backward_(A_lg, μs_f, σs_f2, A_s, Rs)
	
	println("\nVB q(R):")
	show(stdout, "text/plain", R)
	println("\n\nVB q(Q):")
	show(stdout, "text/plain", Q)
	println("\n\nMSE, MAD of VB latent X: ", error_metrics(x_true, μs_s))

	vars = hcat([diag(Σ_s[:, :, t]) for t in 1:T+1]...)
	stds = sqrt.(vars)

	if show_plot
		p = plot(elbos, label = "elbo", title = "ElBO Progression, Hyperparam optim: $hyperoptim")
		plot_file_name = "$(splitext(basename(@__FILE__))[1])_$(Dates.format(now(), "yyyymmdd_HHMMSS")).svg"
		savefig(p, joinpath(expanduser("~/Downloads/_graphs"), plot_file_name))
		display(p)
		sleep(1)

		plots = plot_x_itvl(μs_s, stds, x_true, 20)
		for i in 1:K
			pl = plots[i]
			title!(pl, "VB, hyper-param optim:$t")
			display(pl)
			sleep(1)
		end
	end
	return μs_s, stds, Q_gam
end

function compare_mcmc_vi(mcmc::Vector{T}, vi::Vector{T}) where T
    # Ensure all vectors have the same length
    @assert length(mcmc) == length(vi) "All vectors must have the same length"
    
	p_mcmc = scatter(mcmc, vi, label="MCMC", xlabel = "MCMC", color=:yellow, alpha=0.3)

	p_vi = scatter!(p_mcmc, mcmc, vi, label="VI", ylabel = "VI", color=:blue, alpha=0.3)

	# Determine the range for the y=x line
	min_val = min(minimum(mcmc), minimum(vi))
	max_val = max(maximum(mcmc), maximum(vi))

	# Plot the y=x line
	plot!(p_vi, [min_val, max_val], [min_val, max_val], linestyle=:dash, label = "", color=:red, linewidth=2)

	return p_vi
end

function main(n)
	println("Running experiments for linear growth model:\n")
	println("T = $n\n")
	A_lg = [1.0 1.0; 0.0 1.0]
    C_lg = [1.0 0.0]
	Q = Diagonal([1.0, 1.0])
	R = [0.5]
	K = size(A_lg, 1)
	μ_0 = zeros(K)
	Σ_0 = Diagonal(ones(K))

	println("Ground-truth R:")
	show(stdout, "text/plain", R)
	println("\nGround-truth Q:")
	show(stdout, "text/plain", Q)

	#seeds = [26, 236, 199, 233, 177]
	seeds = [103, 133, 123, 143, 111]

	for sd in seeds
		println("\n----- BEGIN Run seed: $sd -----\n")
		Random.seed!(sd)
		y, x_true = gen_data(A_lg, C_lg, Q, R, μ_0, Σ_0, n)
		comp_vb_mle(y, x_true)
		println("----- END Run seed: $sd -----\n")
	end
end

function plot_mcmc_vi_gamma(a_q, b_q, p_mcmc, true_param = nothing, x_lmin = 0.0, x_lmax = 20.0)
    x_min, x_max = xlims(p_mcmc)
	x_min = max(x_lmin, x_min)
	x_max = min(x_lmax, x_max)
    gamma_dist_q = Gamma(a_q, 1/b_q) 
	ci_lower = quantile(gamma_dist_q, 0.025)
	ci_upper = quantile(gamma_dist_q, 0.975)

    x = range(x_min, x_max, length=100)
    pdf_values = pdf.(gamma_dist_q, x)
    τ_q = plot!(p_mcmc, x, pdf_values, label="VI", lw=2, xlabel="Precision", ylabel="Density")
	
	plot!(τ_q, [ci_lower, ci_upper], [0, 0], line=:stem, marker=:circle, color=:red, label="95% CI", lw=2)
	vspan!(τ_q, [ci_lower, ci_upper], fill=:red, alpha=0.2, label=nothing, xlims=(x_min, x_max))
    
	if true_param !== nothing
		vline!(τ_q, [true_param], label = "ground_truth", linestyle=:dash, linewidth=2)
	end
	return τ_q
end

function comp_vb_mle(y, x_true, hyperoptim=false)
	println("--- MLE ---")
	mle_lg = LocalLinearTrend(vec(y))
	StateSpaceModels.fit!(mle_lg)
	print_results(mle_lg)

	fm = get_filtered_state(mle_lg)
	filter_err = error_metrics(x_true[:, 2:end], fm')

	sm = get_smoothed_state(mle_lg)
	smooth_err = error_metrics(x_true[:, 2:end], sm')

	println("\nMLE Filtered MSE, MAD: ", filter_err)
	println("MLE Smoother MSE, MAD: ", smooth_err)
	println()
	xm_vb, std_vb, Q_gam = test_vb(y, x_true, hyperoptim)
	return xm_vb, std_vb, Q_gam
end

function main_graph(n, sd)
	println("Running experiments for linear growth model (with graph):\n")
	println("T = $n\n")
	A_lg = [1.0 1.0; 0.0 1.0]
    C_lg = [1.0 0.0]
	Q = Diagonal([1.0, 1.0])
	R = [1.0]
	K = size(A_lg, 1)

	println("Ground-truth R:")
	show(stdout, "text/plain", R)
	println("\n\nGround-truth Q:")
	show(stdout, "text/plain", Q)

	println("\n----- BEGIN Run seed: $sd -----\n")
	Random.seed!(sd)
	y, x_true = gen_data(A_lg, C_lg, Q, R, zeros(K), Diagonal(ones(K)), n)

	xm_vb, std_vb, Q_gam = comp_vb_mle(y, x_true)

	xm_mcmc, std_mcmc, p_r, p_q1, p_q2 = test_gibbs(y, x_true, 10000, 5000, 1, true)

	plot_r = plot_mcmc_vi_gamma(Q_gam.a, (Q_gam.b)[1], p_r)
	display(plot_r)

	plot_q1 = plot_mcmc_vi_gamma(Q_gam.α, (Q_gam.β)[1], p_q1)
	display(plot_q1)

	plot_q2 = plot_mcmc_vi_gamma(Q_gam.α, (Q_gam.β)[2], p_q2)
	display(plot_q2)

	p = compare_mcmc_vi(xm_mcmc[1, :], xm_vb[1, 2:end])
	title!(p, "Latent x mean, x_1")
	display(p)

	p_v = compare_mcmc_vi(std_mcmc[1, :], std_vb[1, 2:end])
	title!(p_v, "Latent x std, x_1")
	display(p_v)

	p2 = compare_mcmc_vi(xm_mcmc[2, :], xm_vb[2, 2:end])
	title!(p2, "Latent x mean, x_2")
	display(p2)

	p_2v = compare_mcmc_vi(std_mcmc[2, :], std_vb[2, 2:end])
	#p_2v= qqplot((std_mcmc.^2)[2, :], (std_vb.^2)[2, :], qqline = :R)
	title!(p_2v, "Latent x std, x_2")
	display(p_2v)
	
	println("----- END Run seed: $sd -----\n")
end

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

out_txt(500)

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
