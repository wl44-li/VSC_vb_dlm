include("kl_optim.jl")

begin
	using Distributions, Random
	using LinearAlgebra
	using StatsFuns
	using StatsPlots
	using SpecialFunctions
	using MultivariateStats
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
	D, T = size(ys)
	W_C = ss.W_C
	S_C = ss.S_C
	γ = hps.γ
	a = hps.a
	b = hps.b
	K = length(γ)
	
	# q(ρ), q(C|ρ)
	Σ_C = inv(diagm(γ) + W_C)
	μ_C = [Σ_C * S_C[:, s] for s in 1:D]
	
	G = ys * ys' - S_C' * Σ_C * S_C
	a_ = a + 0.5 * T
	a_s = a_ * ones(D)
    b_s = [b + 0.5 * G[i, i] for i in 1:D]
	
	q_ρ = missing
	
	try
	    q_ρ = Gamma.(a_s, 1 ./ b_s)
	catch err
	    if isa(err, DomainError)
	        println("--- DomainError occurred: check that a_s and b_s are positive values")
			println("\ta_s: ", a_s[1])
			println("\tb_s: ", b_s)
			b_s = abs.(b_s)
			println("\tTemporary fix: ", b_s)
			println("Please consider adjusting hyperparameters α, γ, a, b and re-run PPCA ---\n")

			q_ρ = Gamma.(a_s, 1 ./ b_s)
	    else
	        rethrow(err)
	    end
	end

	ρ̄ = mean.(q_ρ)
	
	# Exp_ϕ 
	Exp_C = S_C'*Σ_C
	Exp_R⁻¹ = diagm(ρ̄)
	Exp_CᵀR⁻¹C = Exp_C'*Exp_R⁻¹*Exp_C + D*Σ_C
	Exp_R⁻¹C = Exp_R⁻¹*Exp_C
	Exp_CᵀR⁻¹ = Exp_C'*Exp_R⁻¹

	# update hyperparameter (after m-step)
	γ_n = [D/((D*Σ_C + Σ_C*S_C*Exp_R⁻¹*S_C'*Σ_C)[j, j]) for j in 1:K]

	# for updating gamma hyperparam a, b       
	exp_ρ = a_s ./ b_s
	exp_log_ρ = [(digamma(a_) - log(b_s[i])) for i in 1:D]
	
	# return expected natural parameters :: Exp_ϕ (for e-step)
	return Exp_ϕ(Exp_C, Exp_R⁻¹, Exp_CᵀR⁻¹C, Exp_R⁻¹C, Exp_CᵀR⁻¹, exp_log_ρ), γ_n, exp_ρ, exp_log_ρ, Qθ(Σ_C, μ_C, a_, b_s)
end

function v_forward(ys::Matrix{Float64}, exp_np::Exp_ϕ, hpp::HPP)
    _, T = size(ys)
    K = length(hpp.γ)

    μs = zeros(K, T)
    Σs = zeros(K, K, T)
	Σs_ = zeros(K, K, T)
	
	# Extract μ_0 and Σ_0 from the HPP struct
    μ_0 = hpp.μ_0
    Σ_0 = hpp.Σ_0

	# initialise for t = 1
	Σ₀_ = Σ_0
	Σs_[:, :, 1] = Σ₀_
	
    Σs[:, :, 1] = inv(I + exp_np.CᵀR⁻¹C)
    μs[:, 1] = Σs[:, :, 1]*(exp_np.CᵀR⁻¹*ys[:, 1])
		
	# iterate over T
	for t in 2:T
		Σₜ₋₁_ = Σs[:, :, t-1]
		Σs_[:, :, t] = Σₜ₋₁_
		
		Σs[:, :, t] = inv(I + exp_np.CᵀR⁻¹C)
    	μs[:, t] = Σs[:, :, t]*(exp_np.CᵀR⁻¹*ys[:, t])
	end

	return μs, Σs, Σs_
end

function log_Z(ys, μs, Σs, Σs_, exp_np::Exp_ϕ, hpp::HPP)
	D, T = size(ys)
	log_Z = 0
	log_det_R = D*log(2π) - sum(exp_np.log_ρ)

	# t = 1
	log_Z += -0.5*(log_det_R - logdet(inv(hpp.Σ_0)*Σs_[:, :, 1]*Σs[:, :, 1]) + hpp.μ_0'*inv(hpp.Σ_0)*hpp.μ_0 - μs[:, 1]'*inv(Σs[:, :, 1])*μs[:, 1] + ys[:, 1]'*exp_np.R⁻¹*ys[:, 1] - transpose(inv(hpp.Σ_0)*hpp.μ_0)*Σs_[:, :, 1]*inv(hpp.Σ_0)*hpp.μ_0)
	
	for t in 2:T
		log_det_Σ = logdet(inv(Σs[:, :, t-1])*Σs_[:, :, t]*Σs[:, :, t])
		μ_t_ = μs[:, t-1]'*inv(Σs[:, :, t-1])*μs[:, t-1]
		μ_t = μs[:, t]'*inv(Σs[:, :, t])*μs[:, t]
		y_t = ys[:, t]'*exp_np.R⁻¹*ys[:, t]
		Σ_μ_t = transpose(inv(Σs[:, :, t-1])*μs[:, t-1])*Σs_[:, :, t]*inv(Σs[:, :, t-1])*μs[:, t-1]

		log_Z += -0.5 * (log_det_R - log_det_Σ + μ_t_ - μ_t + y_t - Σ_μ_t)
	end

	return log_Z
end

function vb_e(ys::Matrix{Float64}, exp_np::Exp_ϕ, hpp::HPP, smooth_out=false)
    _, T = size(ys)
	# forward pass α_t(x_t)
	ωs, Υs, Σs_ = v_forward(ys, exp_np, hpp)
	
	# hidden state sufficient stats 	
	W_C = sum(Υs[:, :, t] + ωs[:, t] * ωs[:, t]' for t in 1:T)
	S_C = sum(ωs[:, t] * ys[:, t]' for t in 1:T)

	if (smooth_out) # return variational smoothed mean, cov of xs, ys after completing VBEM iterations
		return ωs, Υs
	end

	# compute log partition ln Z' (ELBO and convergence check)
	log_Z_ = log_Z(ys, ωs, Υs, Σs_, exp_np, hpp)
	
	return HSS_PPCA(W_C, S_C), log_Z_
end

function update_ab(hpp::HPP, exp_ρ::Vector{Float64}, exp_log_ρ::Vector{Float64})
    D = length(exp_ρ)
    d = mean(exp_ρ)
    c = mean(exp_log_ρ)
    
    # Update `a` using fixed point iteration
	a = hpp.a		

    for _ in 1:1000
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

function vb_ppca(ys::Matrix{Float64}, hpp::HPP, hpp_learn=false, max_iter=300)
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

function vb_ppca_c(ys::Matrix{Float64}, hpp::HPP, hpp_learn=false, max_iter=1000, tol=1e-4)
	D, T = size(ys)
	K = length(hpp.γ)
	
	W_C = Matrix{Float64}(T*I, K, K)
	S_C = Matrix{Float64}(T*I, K, D)

	hss = HSS_PPCA(W_C, S_C)
	exp_np = missing
	elbo_prev = -Inf

	# cf. Beal Algorithm 5.3
	for i in 1:max_iter
		exp_np, γ_n, exp_ρ, exp_log_ρ, qθ = vb_m(ys, hpp, hss)
		hss, log_Z_ = vb_e(ys, exp_np, hpp)

		# Convergence check
		kl_ρ_ = sum([kl_gamma(hpp.a, hpp.b, qθ.a_s, (qθ.b_s)[s]) for s in 1:D])
		kl_C_ = sum([kl_C(zeros(K), hpp.γ, (qθ.μ_C)[s], qθ.Σ_C, exp_ρ[s]) for s in 1:D])
			
		elbo = log_Z_ - kl_ρ_ - kl_C_

		# Hyper-param learning 
		if (hpp_learn)
			if (i%5 == 0) 
				a, b = update_ab(hpp, exp_ρ, exp_log_ρ)
				hpp = HPP(γ_n, a, b, zeros(K), Matrix{Float64}(I, K, K))
			end
		end

		if abs(elbo - elbo_prev) < tol
			println("--- Stopped at iteration: $i ---")
            break
		end
		
        elbo_prev = elbo

		if (i == max_iter)
			println("--- Warning: VB has not necessarily converged at $max_iter iterations ---")
		end
	end
		
	return exp_np, elbo_prev
end

function test_elbo(sd)
	Random.seed!(sd)
	T = 500
	μ_0_t = zeros(2)
	Σ_0_t = Diagonal(ones(2))
	C_ = [1.0 0.0; 0.6 1.0; 0.3 0.2; 0.5 0.1; 0.1 0.3; 0.4 0.1; 0.8 0.2; 0.3 0.3; 0.4 0.6; 0.1 0.4] 
	R_ = Diagonal(ones(10) .* 0.1)
	y_10, x_2 = gen_data(zeros(2, 2), C_, Diagonal([1.0, 1.0]), R_, μ_0_t, Σ_0_t, T)

	elbos = zeros(9)
	for k in 1:9
		γ = ones(k) .* 1000
		a = 0.1
		b = 0.1
		μ_0 = zeros(k)
		Σ_0 = Matrix{Float64}(I, k, k)
		hpp = HPP(γ, a, b, μ_0, Σ_0)
		exp_np, el = vb_ppca_c(y_10, hpp, true)
		println("\nElBO, k=$k :", el)
		elbos[k] = el
		
		if k == 2
			println("K = $k")
			ωs, _, _ = v_forward(y_10, exp_np, hpp)
			println("latent x error: ", error_metrics(x_2, ωs))
			println()

			W = exp_np.C
			WTW = W'*W
			C = WTW + I*mean(diag(inv(exp_np.R⁻¹)))
			y_recon = W*inv(WTW)*C*x_2
			println("y recon error: ", error_metrics(y_10, y_recon))
			println()
		end
	end

	p = plot(elbos, xlabel = "K", ylabel = "ELBO", title = "ELBO comparison", label="")
	vline!(p, [2], label="ground-truth", linestyle=:dash)
	display(p)

	p_ = plot(elbos[1:3], xlabel = "K", ylabel = "ELBO", title = "ELBO comparison", label="")
	vline!(p_, [2], label="ground-truth", linestyle=:dash)
	display(p_)
end

#test_elbo(123)

function comp_ppca(max_T = 1000)
	println("Running experiments for PPCA")
	println("T = $max_T")
	C_ = [1.0 0.0; 0.1 0.8; 0.9 0.2] 
	R = Diagonal([0.5, 0.5, 0.5])
	μ_0 = zeros(2)
	Σ_0 = Matrix{Float64}(I, 2, 2)

	println("Ground-truth Loading Matrix W:")
	show(stdout, "text/plain", C_)
	println("\nGround-truth isotropic co-variance R:")
	show(stdout, "text/plain", R)
	seeds = [103, 133, 123, 105, 233]

	for sd in seeds
		Random.seed!(sd)
		y, x_true = gen_data(zeros(2, 2), C_, Diagonal([1.0, 1.0]), R, μ_0, Σ_0, max_T)
		println("\n----- BEGIN Run seed: $sd -----")
		M_mle = MultivariateStats.fit(PPCA, y; maxoutdim=2) # default MLE
		println("\n--- MLE ---\n Loading Matrix W:")
		show(stdout, "text/plain", loadings(M_mle))
		println("\n\n", M_mle)
		mle_x_pred = MultivariateStats.predict(M_mle, y)
		println("latent x error: ", error_metrics(x_true, mle_x_pred))
		mle_y_recon = MultivariateStats.reconstruct(M_mle, x_true)
		println("reconstruction y error: ", error_metrics(y, mle_y_recon))

		M_em = MultivariateStats.fit(PPCA, y; method=(:em), maxoutdim=2)
		println("\n--- EM ---\n Loading Matrix W:")
		show(stdout, "text/plain", loadings(M_em))
		println("\n\n", M_em)
		em_x_pred = MultivariateStats.predict(M_em, y)
		println("latent x error: ", error_metrics(x_true, em_x_pred))
		em_y_recon = MultivariateStats.reconstruct(M_em, x_true)
		println("reconstruction y error: ", error_metrics(y, em_y_recon))

		M_bay = MultivariateStats.fit(PPCA, y; method=(:bayes), maxoutdim=2)
		println("\n--- Bayes ---\n Loading Matrix W:")
		show(stdout, "text/plain", loadings(M_bay))
		println("\n\n", M_bay)
		bay_x_pred = MultivariateStats.predict(M_bay, y)
		println("latent x error: ", error_metrics(x_true, bay_x_pred))
		bay_y_recon = MultivariateStats.reconstruct(M_bay, x_true)
		println("reconstruction y error: ", error_metrics(y, bay_y_recon))

		hpp = HPP(ones(2) .* 10, 0.1, 0.1, μ_0, Σ_0)
		println("\n--- VBEM ---\n Loading Matrix W:")
		exp_np, _ = vb_ppca_c(y, hpp, true)
		show(stdout, "text/plain", exp_np.C)
		println("\nIsotropic Co-variance R: ")
		show(stdout, "text/plain", inv(exp_np.R⁻¹))
		ωs, _, _ = v_forward(y, exp_np, hpp)
		println("\nlatent x error: ", error_metrics(x_true, ωs))
		W = exp_np.C
		WTW = W'*W
		C = WTW + I*mean(diag(inv(exp_np.R⁻¹)))
		y_recon = W*inv(WTW)*C*x_true
		println("y recon error: ", error_metrics(y, y_recon))
		println("----- END Run seed: $sd -----\n")
	end
end

#comp_ppca(1000)

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

"""
TO-DO: Compare ELBO for optimal K, convex optimisation? VI always reaches the same ELBO
"""

function k_elbo_p3(y, n=10)
	elbos_k1 = zeros(n)
	elbos_k2 = zeros(n)  

	index_1 = 1
	index_2 = 1

	for _ in 1:n

		for k in 1:2
			γ = ones(k) 
			a = 0.5
			b = 0.5
			μ_0 = zeros(k)
			Σ_0 = Matrix{Float64}(I, k, k)
			hpp = HPP(γ, a, b, μ_0, Σ_0)
			_, el = vb_ppca_c(y, hpp, true)

			if k == 1
				elbos_k1[index_1] = el
				index_1+=1
			end

			if k == 2
				elbos_k2[index_2] = el
				index_2+=1
			end
		end
	end

	println(elbos_k1)
	println(elbos_k2)

	groups = repeat(["K = 1", "K = 2"], inner = length(elbos_k1))
	all_elbos = vcat(elbos_k1, elbos_k2)

	p2 = dotplot(groups, all_elbos, group=groups, color=[:blue :orange], label="", ylabel="ELBO", legend=false)
	title!(p2, "ELBO Model Selection, K=2")
    display(p2)
end

function main()
	# P = 3, K = 2
	C_ = [1.0 0.0; 0.1 0.8; 0.9 0.2]
	σ² = 0.5
	R = Diagonal(ones(3) .* σ²)

	println("Ground-truth Loading Matrix W:")
	show(stdout, "text/plain", C_)
	println("\nσ²: ", σ²)
	y, _ = gen_data(zeros(2, 2), C_, Diagonal([1.0, 1.0]), R, zeros(2), Diagonal([1.0, 1.0]), 500)

	k_elbo_p3(y)
end

main()

#out_txt(500)

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