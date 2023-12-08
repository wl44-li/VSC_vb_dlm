include("kl_optim.jl")
include("turing_ppca.jl")
begin
	using Distributions, Random
	using LinearAlgebra
	using StatsFuns
	using StatsPlots
	using SpecialFunctions
	using MultivariateStats
end

function gen_data(A, C, Q, R, μ_0, Σ_0, T)
	K = size(A, 1)
	D = size(C, 1)
	x = zeros(K, T)
	y = zeros(D, T)

	x[:, 1] = rand(MvNormal(A*μ_0, Q))
	y[:, 1] = C * x[:, 1] + rand(MvNormal(zeros(D), R))

	for t in 2:T
		x[:, t] = A * x[:, t-1] + rand(MvNormal(zeros(K), Q))
		y[:, t] = C * x[:, t] + rand(MvNormal(zeros(D), R)) 
	end

	return y, x
end

begin
	struct HSS_PPCA
	    W_C::Matrix{Float64}
	    S_C::Matrix{Float64}
	end

	struct Exp_ϕ
		C
		R⁻¹
		CᵀR⁻¹C
		R⁻¹C
		CᵀR⁻¹
		log_ρ
	end

	struct HPP
	    γ::Vector{Float64}  # precision vector for emission C
	    a::Float64 # gamma rate of ρ
	    b::Float64 # gamma inverse scale of ρ
	    μ_0::Vector{Float64} # auxiliary hidden state mean
	    Σ_0::Matrix{Float64} # auxiliary hidden state co-variance
	end

	struct Qθ
		Σ_C # q(C)
		μ_C # q(C)
		a_s # q(ρ)
		b_s # q(ρ)
	end
end

function vb_m(ys, hps::HPP, ss::HSS_PPCA)
	P, T = size(ys)
	W_C = ss.W_C
	S_C = ss.S_C
	γ = hps.γ
	a = hps.a
	b = hps.b
	K = length(γ)
	
	# q(C|ρ)
	Σ_C = inv(diagm(γ) + W_C)
	μ_C = [Σ_C * S_C[:, s] for s in 1:P]
	
	# q(ρ) 
	G = ys * ys' - S_C' * Σ_C * S_C
	a_s = a + 0.5 * T * P
    b_s = b + 0.5 * tr(G)
	q_ρ = Gamma(a_s, 1 / b_s)
	ρ̄ = mean(q_ρ)
	
	# Exp_ϕ 
	Exp_C = S_C'*Σ_C
	Exp_R⁻¹ = diagm(ones(P) .* ρ̄)
	Exp_CᵀR⁻¹C = Exp_C'*Exp_R⁻¹*Exp_C + P*Σ_C
	Exp_R⁻¹C = Exp_R⁻¹*Exp_C
	Exp_CᵀR⁻¹ = Exp_C'*Exp_R⁻¹

	# update hyperparameter (after m-step)
	γ_n = [P/((P*Σ_C + Σ_C*S_C*Exp_R⁻¹*S_C'*Σ_C)[j, j]) for j in 1:K]

	# for updating gamma hyperparam a, b       
	exp_ρ = ones(P) .* (a_s / b_s)
	exp_log_ρ = [(digamma(a_s) - log(b_s)) for _ in 1:P]
	
	# return expected natural parameters :: Exp_ϕ (for e-step)
	return Exp_ϕ(Exp_C, Exp_R⁻¹, Exp_CᵀR⁻¹C, Exp_R⁻¹C, Exp_CᵀR⁻¹, exp_log_ρ), γ_n, exp_ρ, exp_log_ρ, Qθ(Σ_C, μ_C, a_s, b_s)
end

function v_forward(ys, exp_np::Exp_ϕ, hpp::HPP)
    _, T = size(ys)
    K = length(hpp.γ)
    μs = zeros(K, T)
    Σs = zeros(K, K, T)

	# iterate over T, Algorithm 5.1
	for t in 1:T		
		Σs[:, :, t] = inv(I + exp_np.CᵀR⁻¹C)
    	μs[:, t] = Σs[:, :, t]*(exp_np.CᵀR⁻¹*ys[:, t])
	end

	return μs, Σs
end

"""
Simplified for PPCA, double check with Algorithm 5.1
"""
function log_Z(ys, μs, Σs, exp_np::Exp_ϕ, hpp::HPP)
	D, T = size(ys)
	log_Z = 0
	log_det_R = D*log(2π) - sum(exp_np.log_ρ)

	# t = 1
	log_Z += -0.5*(log_det_R - logdet(Σs[:, :, 1]) + hpp.μ_0'*inv(hpp.Σ_0)*hpp.μ_0 - μs[:, 1]'*inv(Σs[:, :, 1])*μs[:, 1] + ys[:, 1]'*exp_np.R⁻¹*ys[:, 1] - transpose(inv(hpp.Σ_0)*hpp.μ_0)*hpp.μ_0)
	
	for t in 2:T
		log_det_Σ = logdet(Σs[:, :, t])
		μ_t_ = μs[:, t-1]'*inv(Σs[:, :, t-1])*μs[:, t-1]
		μ_t = μs[:, t]'*inv(Σs[:, :, t])*μs[:, t]
		y_t = ys[:, t]'*exp_np.R⁻¹*ys[:, t]
		Σ_μ_t = transpose(inv(Σs[:, :, t-1])*μs[:, t-1])*μs[:, t-1]

		log_Z += -0.5 * (log_det_R - log_det_Σ + μ_t_ - μ_t + y_t - Σ_μ_t)
	end

	return log_Z
end

function vb_e(ys, exp_np::Exp_ϕ, hpp::HPP, smooth_out=false)
    _, T = size(ys)
	# forward pass α_t(x_t) suffices as A = 0 matrix
	ωs, Υs = v_forward(ys, exp_np, hpp)
	
	# hidden state sufficient stats 	
	W_C = sum(Υs[:, :, t] + ωs[:, t] * ωs[:, t]' for t in 1:T)
	S_C = sum(ωs[:, t] * ys[:, t]' for t in 1:T)

	if (smooth_out) # return variational smoothed mean, cov of xs, ys after completing VBEM iterations
		return ωs, Υs
	end

	# compute log partition ln Z' (ELBO and convergence check)
	log_Z_ = log_Z(ys, ωs, Υs, exp_np, hpp)
	
	return HSS_PPCA(W_C, S_C), log_Z_
end

function update_ab(hpp::HPP, exp_ρ::Vector{Float64}, exp_log_ρ::Vector{Float64})
    d = exp_ρ[1]
    c = exp_log_ρ[1]
    
    # Update `a` using fixed point iteration
	a = hpp.a		

    for _ in 1:10
        ψ_a = digamma(a)
        ψ_a_p = trigamma(a)
        
        a_new = a * exp(-(ψ_a - log(a) + log(d) - c) / (a * ψ_a_p - 1))
		a = a_new

		# check convergence
        if abs(a_new - a) < 1e-4
            break
        end
    end
    
    # Update `b` using the converged value of `a`
    b = a/d

	return a, b
end

function vb_ppca(ys, hpp::HPP, hpp_learn=false, max_iter=500)
	D, T = size(ys)
	K = length(hpp.γ)
	
	# no random initialistion
	W_C = Matrix{Float64}(T*I, K, K)
	S_C = Matrix{Float64}(T*I, K, D)
	
	hss = HSS_PPCA(W_C, S_C)
	exp_np = missing

	for i in 1:max_iter
		exp_np, γ_n, exp_ρ, exp_log_ρ, _ = vb_m(ys, hpp, hss)
		
		hss, _ = vb_e(ys, exp_np, hpp)

		if (hpp_learn)
			if (i%5 == 0) # update hyperparam every 5 iterations
				a, b = update_ab(hpp, exp_ρ, exp_log_ρ)
				hpp = HPP(γ_n, a, b, μ_0, Σ_0)
			end
		end
	end
		
	return exp_np
end

function vb_ppca_c(ys, hpp::HPP, hpp_learn=false, max_iter=1000, tol=1e-4; init="mle", debug=false)
	P, _ = size(ys)
	K = length(hpp.γ)
	hss = missing
	qθ = missing

	if init == "mle"
		# use MLE esitmate of σ and C to run VBE-step first
		"""
		Warning: Work in progress
		"""

		println("\t--- VB PPCA using MLE initilaization ---")
		M_mle = MultivariateStats.fit(PPCA, ys; maxoutdim=K)

		σ²_init = M_mle.σ² .* (1 + randn() * 0.3) 
		e_C = M_mle.W[:, 1:K] * (1 + randn() * 0.3) 

		R = diagm(ones(P) .* σ²_init)
		e_R⁻¹ = inv(R)
		e_CᵀR⁻¹C = e_C'*e_R⁻¹*e_C
		e_R⁻¹C = e_R⁻¹*e_C
		e_CᵀR⁻¹ = e_C'*e_R⁻¹
		e_log_ρ = log.(1 ./ diag(R))
		exp_np = Exp_ϕ(e_C, e_R⁻¹, e_CᵀR⁻¹C, e_R⁻¹C, e_CᵀR⁻¹, e_log_ρ)
		
		hss, _ = vb_e(ys, exp_np, hpp)
		if debug
			println("--- Init PPCA with MLE ---")
			println("C_init: ", e_C)
			println("R_init: ", R)
			println("w_c, s_c :", hss)
		end
	end

	if init == "em"
		# use EM esitmate of σ and C to run VBE-step first
		"""
			Warning: Work in progress
		"""
		M_em = MultivariateStats.fit(PPCA, y; method=(:em), maxoutdim=2)

		σ²_init = M_em.σ² .* (1 + randn() * 0.2) 
		e_C = M_em.W * (1 + randn() * 0.2) 

		R = diagm(ones(P) .* σ²_init)
		e_R⁻¹ = inv(R)
		e_CᵀR⁻¹C = e_C'*e_R⁻¹*e_C
		e_R⁻¹C = e_R⁻¹*e_C
		e_CᵀR⁻¹ = e_C'*e_R⁻¹
		e_log_ρ = log.(1 ./ diag(R))
		exp_np = Exp_ϕ(e_C, e_R⁻¹, e_CᵀR⁻¹C, e_R⁻¹C, e_CᵀR⁻¹, e_log_ρ)
		
		hss, _ = vb_e(ys, exp_np, hpp)
		if debug
			println("--- Init PPCA with EM ---")
			println("C_init: ", e_C)
			println("R_init: ", R)
			println("w_c, s_c :", hss)
		end
	end

	if init == "fixed"
		W_C = Matrix{Float64}(P*I, K, K)
		S_C = Matrix{Float64}(P*I, K, P)
		hss = HSS_PPCA(W_C, S_C)

		if debug
			println("w_c, s_c :", hss)
		end
	end

	if init == "random"
		# init R and C, needs more testing
		println("\t--- VB PPCA using random initilaization ---")
		ρ_init = 0.5 / 0.5
		#ρ_init = hpp.a / hpp.b

		R = diagm(ones(P) .* ρ_init)

		cs = [rand(MvNormal(zeros(K), (1/ρ_init) * I)) for _ in 1:P]
		e_C = (hcat(cs...))'
		
		e_R⁻¹ = inv(R)
		e_CᵀR⁻¹C = e_C'*e_R⁻¹*e_C
		e_R⁻¹C = e_R⁻¹*e_C
		e_CᵀR⁻¹ = e_C'*e_R⁻¹
		e_log_ρ = log.(1 ./ diag(R))
		exp_np = Exp_ϕ(e_C, e_R⁻¹, e_CᵀR⁻¹C, e_R⁻¹C, e_CᵀR⁻¹, e_log_ρ)
		
		hss, _ = vb_e(ys, exp_np, hpp)
		if debug
			println("--- Init PPCA Random ---")
			println("C_init: ", e_C)
			println("R_init: ", R)
			println("w_c, s_c, :", hss)
		end
	end
	
	exp_np = missing
	elbo_prev = -Inf
	el_s = zeros(max_iter)

	# cf. Beal Algorithm 5.3
	for i in 1:max_iter
		exp_np, γ_n, exp_ρ, exp_log_ρ, qθ = vb_m(ys, hpp, hss)
		hss, log_Z_ = vb_e(ys, exp_np, hpp)

		# Convergence check
		kl_ρ_ = sum([kl_gamma(hpp.a, hpp.b, qθ.a_s, (qθ.b_s)) for _ in 1:P])
		kl_C_ = sum([kl_C(zeros(K), hpp.γ, (qθ.μ_C)[s], qθ.Σ_C, exp_ρ[s]) for s in 1:P])
		elbo = log_Z_ - kl_ρ_ - kl_C_
		el_s[i] = elbo

		if debug
			println("\nVB iter $i: ")
			println("\tR: ", inv(exp_np.R⁻¹))
			println("\tα_r, β_r: ", qθ.a_s, qθ.b_s)
			println("\tLog Z: ", log_Z_)
			println("\tKL r (gamma): ", kl_ρ_)
			println("\tKL C (Normal): ", kl_C_)
			println("\tElbo $i: ", elbo)
		end

		# Hyper-param learning, γ, a, b
		if (hpp_learn)
			if (i%5 == 0) 
				a, b = update_ab(hpp, exp_ρ, exp_log_ρ)
				hpp = HPP(γ_n, a, b, hpp.μ_0, hpp.Σ_0)
			end
		end

		if abs(elbo - elbo_prev) < tol
			println("--- Stopped at iteration: $i ---")
			el_s = el_s[1:i]
            break
		end
		
		if (i == max_iter)
			println("--- Warning: VB has not necessarily converged at $max_iter iterations, 
			last elbo difference: $(abs(elbo-elbo_prev)), tolerance: $tol ---")
		end

		elbo_prev = elbo
	end
		
	if debug
		p = plot(el_s, title="Debug ELBO progression")
		display(p)
	end
	return exp_np, elbo_prev, el_s, qθ
end

function comp_ppca(max_T = 1000)
	println("Running experiments for PPCA")
	println("T = $max_T")
	C_ = [1.0 0.0; 0.2 1.0; 0.9 0.1] 
	σ² = 3.0
	R = Diagonal(ones(3) .* σ²)
	μ_0 = zeros(2)
	Σ_0 = Matrix{Float64}(I, 2, 2)

	println("Ground-truth Loading Matrix W:")
	show(stdout, "text/plain", C_)
	println("\nGround-truth isotropic co-variance R:")
	show(stdout, "text/plain", R)

	y, x_true = gen_data(zeros(2, 2), C_, Diagonal([1.0, 1.0]), R, μ_0, Σ_0, max_T)
	
	M_mle = MultivariateStats.fit(PPCA, y; maxoutdim=2) # default MLE
	println("\n--- MLE ---\n Loading Matrix W:")
	show(stdout, "text/plain", loadings(M_mle))
	println("\n\n", M_mle)
	mle_x_pred = MultivariateStats.predict(M_mle, y)
	println("latent x MSE, MAD: ", error_metrics(x_true, mle_x_pred))
	mle_y_recon = MultivariateStats.reconstruct(M_mle, mle_x_pred)
	println("reconstruction y MSE, MAD: ", error_metrics(y, mle_y_recon))

	M_em = MultivariateStats.fit(PPCA, y; method=(:em), maxoutdim=2)
	println("\n--- EM ---\n Loading Matrix W:")
	show(stdout, "text/plain", loadings(M_em))
	println("\n\n", M_em)
	em_x_pred = MultivariateStats.predict(M_em, y)
	println("latent x MSE, MAD: ", error_metrics(x_true, em_x_pred))
	em_y_recon = MultivariateStats.reconstruct(M_em, x_true)
	println("reconstruction y MSE, MAD: ", error_metrics(y, em_y_recon))

	M_bay = MultivariateStats.fit(PPCA, y; method=(:bayes), maxoutdim=2)
	println("\n--- Bayes ---\n Loading Matrix W:")
	show(stdout, "text/plain", loadings(M_bay))
	println("\n\n", M_bay)
	bay_x_pred = MultivariateStats.predict(M_bay, y)
	println("latent x error: ", error_metrics(x_true, bay_x_pred))
	bay_y_recon = MultivariateStats.reconstruct(M_bay, bay_x_pred)
	println("reconstruction y error: ", error_metrics(y, bay_y_recon))

	hpp = HPP(ones(2), 2, 1e-4, zeros(2), Matrix{Float64}(I, 2, 2))
	println("\n--- VBEM ---\n Loading Matrix W:")
	exp_np, _ , els = vb_ppca_c(y, hpp, false, init="fixed")
	show(stdout, "text/plain", exp_np.C)
	println("\nIsotropic Co-variance R: ")
	show(stdout, "text/plain", inv(exp_np.R⁻¹))
	ωs, _, _ = v_forward(y, exp_np, hpp)
	println("\nlatent x MSE, MAD: ", error_metrics(x_true, ωs))
	W = exp_np.C
	WTW = W'*W
	C = WTW + I*(inv(exp_np.R⁻¹))
	y_recon = W*inv(WTW)*C*ωs
	println("y recon MSE, MAD: ", error_metrics(y, y_recon))
	p = plot(els, title="ELBO progression", label="")
	display(p)
end

#comp_ppca()

function out_txt(n)
	file_name = "$(splitext(basename(@__FILE__))[1])_$(Dates.format(now(), "yyyymmdd_HHMMSS")).txt"

	open(file_name, "w") do f
		redirect_stdout(f) do
			redirect_stderr(f) do
				comp_ppca(n)
			end
		end
	end
end

function k_elbo_p4(y, n=10, hyper_optim=false; verboseOut=false)
	elbos_k1 = zeros(n)
	elbos_k2 = zeros(n)  
	elbos_k3 = zeros(n)
	p_elbo_k1 = plot()
	p_elbo_k2 = plot()
	p_elbo_k3 = plot()

	title!(p_elbo_k1, "K=1")
	title!(p_elbo_k2, "K=2")
	title!(p_elbo_k3, "K=3")

	index_1 = 1
	index_2 = 1
	index_3 = 1

	for _ in 1:n
		for k in 1:3
			#γ = ones(k) * 1e-5
			γ = ones(k)
			a = 2
			b = 1e-4
			μ_0 = zeros(k)
			Σ_0 = Matrix{Float64}(I, k, k)
			hpp = HPP(γ, a, b, μ_0, Σ_0)

			exp_end, el_end, els = vb_ppca_c(y, hpp, hyper_optim, init="mle")

			if k == 1
				plot!(p_elbo_k1, els, label="", ylabel="ElBO", xlabel="Iterations")
				elbos_k1[index_1] = el_end
				index_1+=1

				if verboseOut
					println("\n--- VBEM k=$k ---\n Loading Matrix W:")
					show(stdout, "text/plain", exp_end.C)
					println("\nIsotropic noise ", inv(exp_end.R⁻¹)[1, 1])
				end
			end

			if k == 2
				plot!(p_elbo_k2, els, label="", ylabel="ElBO", xlabel="Iterations")
				
				if verboseOut
					println("\n--- VBEM k=$k ---\n Loading Matrix W:")
					show(stdout, "text/plain", exp_end.C)
					println("\nIsotropic noise ", inv(exp_end.R⁻¹)[1, 1])
				end
				elbos_k2[index_2] = el_end
				index_2+=1
			end

			if k == 3
				plot!(p_elbo_k3, els, label="", ylabel="ElBO", xlabel="Iterations")
				elbos_k3[index_3] = el_end
				index_3+=1
			end
		end
	end

	if verboseOut
		println("final ElBO of K=1:", elbos_k1)
		println("final ElBO of K=2:", elbos_k2)
		println("final ElBO of K=3:", elbos_k3)
	end

	println("--- FINAL ElBO mean: ---")
	println("\tK=1:", mean(elbos_k1))
	println("\tK=2:", mean(elbos_k2))
	println("\tK=3:", mean(elbos_k3))

	display(p_elbo_k1)
	display(p_elbo_k2)
	display(p_elbo_k3)

	groups = repeat(["K = 1", "K = 2", "K = 3"], inner = length(elbos_k1))
	all_elbos = vcat(elbos_k1, elbos_k2, elbos_k3)

	p2 = dotplot(groups, all_elbos, group=groups, color=[:blue :orange :green], label="", ylabel="ElBO", legend=false)
	title!(p2, "ElBO Model Selection, K=2")
    display(p2)
end

function compare_mcmc_vi(mcmc::Vector{T}, vi::Vector{T}) where T
    # Ensure all vectors have the same length
    @assert length(mcmc) == length(vi) "All vectors must have the same length"
    
	p_mcmc = scatter(mcmc, vi, label="", color=:red, ms=2, alpha=0.5)
	p_vi = scatter!(p_mcmc, mcmc, vi, label="", ylabel = "VI", ms=2, color=:green, alpha=0.5)

	# Determine the range for the y=x line
	min_val = min(minimum(mcmc), minimum(vi))
	max_val = max(maximum(mcmc), maximum(vi))

	# Plot the y=x line
	plot!(p_vi, [min_val, max_val], [min_val, max_val], ls=:dash, label = "", color=:blue, lw=2)

	return p_vi
end

function test_vb_trivial()
	T = 500
    C_d2k1 = reshape([1.0, 0.5], 2, 1)
    R_2 = Diagonal([1.0, 1.0])
    y, _ = gen_data_ppca([0.0], C_d2k1, [1.0], R_2, T)

	K = 1
	γ = ones(K)
	a = 2
	b = 1e-3
	μ_0 = zeros(K)
	Σ_0 = Matrix{Float64}(I, K, K)
	hpp = HPP(γ, a, b, μ_0, Σ_0)
	exp_np, _, _, qθ = vb_ppca_c(y, hpp, false, init="mle")

	return exp_np, qθ
end

#exp_np, qθ = test_vb_trivial()

function comp_mcmc_vb(mcmc_mode="hmc")
	T = 500
    C_d2k1 = reshape([1.0, 0.5], 2, 1)
    R_2 = Diagonal([1.0, 1.0])
    y, x_true = gen_data_ppca([0.0], C_d2k1, [1.0], R_2, T)

	mcmc_chain = missing
	if mcmc_mode == "hmc"
		mcmc_chain = hmc_ppca(y, 1)
	end

	if mcmc_mode == "nuts"
		mcmc_chain = nuts_ppca_alt(y, 1)
	end

	K = 1
	γ = ones(K)
	a = 2
	b = 1e-3
	μ_0 = zeros(K)
	Σ_0 = Matrix{Float64}(I, K, K)
	hpp = HPP(γ, a, b, μ_0, Σ_0)

	exp_np, _, _, qθ = vb_ppca_c(y, hpp, false, init="random")

	return mcmc_chain, qθ, exp_np, x_true, y
end

# hmc_chain, _, exp_np, x_true, y = comp_mcmc_vb()
# T = 500
# x_means = Vector{Float64}(undef, T)
# x_stds = Vector{Float64}(undef, T)

# for t in 1:T
# 	samples = hmc_chain[Symbol("x[1,$t]")].data
# 	x_means[t] = mean(samples)
# 	x_stds[t] = std(samples)
# end

# K = 1
# γ = ones(K)
# a = 2
# b = 1e-3
# μ_0 = zeros(K)
# Σ_0 = Matrix{Float64}(I, K, K)
# hpp = HPP(γ, a, b, μ_0, Σ_0)

# ωs, Σs = v_forward(y, exp_np, hpp)

# p_mean = compare_mcmc_vi(abs.(x_means), abs.(vec(ωs)))
# xlabel!(p_mean, "MCMC (ground-truth)")
# display(p_mean)

# p_std = compare_mcmc_vi(x_stds, vec(sqrt.(Σs)))
# #xlims!(p_std, 0.65, 0.75)
# xlabel!(p_std, "MCMC (ground-truth)")
# display(p_std)

function gen_plots(mode="hmc")
	mcmc_chain_k1, qθ_k1 = comp_mcmc_vb(mode)

	τs = mcmc_chain_k1[:τ].data
	p_t = density(τs, label = "MCMC($mode)", lw=2)
	gamma_dist_r = InverseGamma(qθ_k1.a_s, qθ_k1.b_s)
	xs = range(0.8, 1.2, length=100)

	ci_lower = quantile(gamma_dist_r, 0.025)
	ci_upper = quantile(gamma_dist_r, 0.975)

	pdf_values = pdf.(gamma_dist_r, xs)
	plot!(p_t, xs, pdf_values, label="VI", lw=2, ylabel="Density")
	plot!(p_t, [ci_lower, ci_upper], [0, 0], line=:stem, marker=:circle, ms=2, color=:red, label="95% CI", lw=2)
	vspan!(p_t, [ci_lower, ci_upper], fill=:red, alpha=0.1, label=nothing)
	xlabel!(p_t, "σ²")
	display(p_t)

	c1s, c2s = mcmc_chain_k1[Symbol("C[1,1]")].data, mcmc_chain_k1[Symbol("C[1,2]")].data
	p1 = density(abs.(c1s), label = "MCMC($mode)", lw=2)
	norm_c_1 = Normal(abs.(qθ_k1.μ_C[1])[1], sqrt.(qθ_k1.Σ_C)[1])
	xs = range(0.8, 1.3, length=100)
	pdf_values = pdf.(norm_c_1, xs)
	plot!(p1, xs, pdf_values, label="VI", lw=2, ylabel="Density")
	ci_lower = quantile(norm_c_1, 0.025)
	ci_upper = quantile(norm_c_1, 0.975)
	plot!(p1, [ci_lower, ci_upper], [0, 0], line=:stem, marker=:circle, ms=2, color=:red, label="95% CI", lw=2)
	vspan!(p1, [ci_lower, ci_upper], fill=:red, alpha=0.1, label=nothing)
	xlabel!(p1, "C[1, 1]")
	display(p1)

	p2 = density(abs.(c2s), label = "MCMC($mode)", lw=2)
	norm_c_2 = Normal(abs.(qθ_k1.μ_C[2])[1], sqrt.(qθ_k1.Σ_C)[1])
	xs = range(0.3, 0.8, length=100)
	pdf_values = pdf.(norm_c_2, xs)
	plot!(p2, xs, pdf_values, label="VI", lw=2, ylabel="Density")
	ci_lower = quantile(norm_c_2, 0.025)
	ci_upper = quantile(norm_c_2, 0.975)
	plot!(p2, [ci_lower, ci_upper], [0, 0], line=:stem, marker=:circle, ms=2, color=:red, label="95% CI", lw=2)
	vspan!(p2, [ci_lower, ci_upper], fill=:red, alpha=0.1, label=nothing)
	xlabel!(p2, "C[2, 1]")
	display(p2)
end

#gen_plots()
#gen_plots("nuts")

function main(n)
	# P = 4, K = 2 truth
	C_ = [1.0 0.0; 0.5 1.0; 0.3 0.8; 0.9 0.1]
	σ² = 5.0
	R = Diagonal(ones(4) .* σ²)

	println("Ground-truth\nLoading Matrix W:")
	show(stdout, "text/plain", C_)
	println("\nIsotropic noise σ²: ", σ²)
	Random.seed!(10)
	y, _ = gen_data(zeros(2, 2), C_, Diagonal([1.0, 1.0]), R, zeros(2), Diagonal(ones(2)), n)
	k_elbo_p4(y, 10, false, verboseOut=false)
end

#main(5000)

# for MNIST data
function vb_ppca_k2(y, em_iter=100, hp_optim=false; debug=false, mode="mle")
	
	# related to row-precision of C/W
	γ = ones(2) 

	# related to precision of σ²
	# a = 1.1
	# b = 0.05

	a = 2
	b = 1e-3

	μ_0 = zeros(2)
	Σ_0 = Matrix{Float64}(I, 2, 2)
	hpp = HPP(γ, a, b, μ_0, Σ_0)

	@time exp_np, _, els = vb_ppca_c(y, hpp, hp_optim, em_iter, init=mode)

	xs, xs_var = v_forward(y, exp_np, hpp)

	if debug
		#println("ELBOs: ", els)
		println("ELBOs end: ", els[end])
		p_el = plot(els, label="ElBO")
		display(p_el)
	end

	# return matrix of factor loadings (W), and isotropic noise (σ²)
	return exp_np.C, exp_np.R⁻¹[1, 1], els, xs, xs_var
end

# PLUTO_PROJECT_TOML_CONTENTS = """
# [deps]
# Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
# LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
# MultivariateStats = "6f286f6a-111f-5878-ab1e-185364afe411"
# Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
# PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
# Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
# SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
# StatsFuns = "4c63d2b9-4356-54db-8cca-17b64c39e42c"

# [compat]
# Distributions = "~0.25.96"
# MultivariateStats = "~0.10.2"
# Plots = "~1.38.16"
# PlutoUI = "~0.7.51"
# SpecialFunctions = "~2.2.0"
# StatsFuns = "~1.3.0"
# """