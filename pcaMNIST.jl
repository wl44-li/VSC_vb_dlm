# import Pkg
# Pkg.add("MLDatasets")
# Pkg.add("Measures")
# Pkg.add("PyPlot")
include("vb_ppca.jl")
using MultivariateStats
using MLDatasets: MNIST
using LinearAlgebra
using Measures
using Plots
using StatsBase
#pyplot()

"""
BEALE VBEM
"""
struct Exp_ϕ_B
	A
	AᵀA
	C
	R⁻¹
	CᵀR⁻¹C
	R⁻¹C
	CᵀR⁻¹
	log_ρ
end

struct HPP_B
    α::Vector{Float64} # precision vector for transition A
    γ::Vector{Float64}  # precision vector for emission C
    a::Float64 # gamma rate of ρ
    b::Float64 # gamma inverse scale of ρ
    μ_0::Vector{Float64} # auxiliary hidden state mean
    Σ_0::Matrix{Float64} # auxiliary hidden state co-variance
end

function kl_A(μ_0, Σ_0, μ_A, Σ_A)
	Σ_inv = inv(Σ_0)
	kl = -0.5*logdet(Σ_A*Σ_inv)
	kl -= 0.5*tr(I - (Σ_A + (μ_A - μ_0)*(μ_A - μ_0)')*Σ_inv)
	return kl
end

function kl_ρ(a_0, b_0, a_s, b_s)
	kl = a_s*log(b_s) - a_0*log(b_0) - loggamma(a_s) + loggamma(a_0)
	kl += (a_s - a_0)*(digamma(a_s) - log(b_s))
	kl -= a_s*(1 - b_0/b_s)
	return kl
end

function kl_C(μ_0, γ, μ_C, Σ_C, exp_ρs)
	kl = -0.5*logdet(Σ_C*Diagonal(γ))
	kl -= 0.5*tr(I - (Σ_C*Diagonal(γ) + (μ_C - μ_0)*(μ_C - μ_0)')*exp_ρs*Diagonal(γ))
	return kl
end

function log_Z(ys, μs, Σs, Σs_, exp_np::Exp_ϕ_B, hpp::HPP_B)
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

struct Qθ_B
	Σ_A # q(A)
	μ_A # q(A)
	Σ_C # q(C)
	μ_C # q(C)
	a_s # q(ρ)
	b_s # q(ρ)
end

function vb_m_b(ys, hps::HPP_B, ss::HSS)
	D, T = size(ys)
	W_A = ss.W_A
	S_A = ss.S_A
	W_C = ss.W_C
	S_C = ss.S_C
	α = hps.α
	γ = hps.γ
	a = hps.a
	b = hps.b
	K = length(α)
	
	# q(A), q(ρ), q(C|ρ)
    Σ_A = inv(diagm(α) + W_A)
	Σ_C = inv(diagm(γ) + W_C)

	μ_A = [Σ_A * S_A[:, j] for j in 1:K]
	μ_C = [Σ_C * S_C[:, s] for s in 1:D]
	
	# q(ρ), isotropic noise
	G = ys * ys' - S_C' * Σ_C * S_C
	a_s = a + 0.5 * T * D
    b_s = b + 0.5 * tr(G)
	q_ρ = Gamma(a_s, 1 / b_s)
	ρ̄ = mean(q_ρ)
	
	# Exp_ϕ 
	Exp_A = [0.0 1.0; 1.0 0.0]
	Exp_AᵀA = Exp_A'*Exp_A + K*Σ_A
	Exp_C = S_C'*Σ_C
	Exp_R⁻¹ = diagm(ones(D) .* ρ̄)
	
	Exp_CᵀR⁻¹C = Exp_C'*Exp_R⁻¹*Exp_C + D*Σ_C
	Exp_R⁻¹C = Exp_R⁻¹*Exp_C
	Exp_CᵀR⁻¹ = Exp_C'*Exp_R⁻¹

	# update hyperparameter for A, C row priors
	α_n = [K/((K*Σ_A + Σ_A*S_A*S_A'*Σ_A)[j, j]) for j in 1:K]
	γ_n = [D/((D*Σ_C + Σ_C*S_C*Exp_R⁻¹*S_C'*Σ_C)[j, j]) for j in 1:K]

	# for updating gamma hyperparam a, b, R priors       
	exp_ρ = ones(D) .* (a_s / b_s)
	exp_log_ρ = [(digamma(a_s) - log(b_s)) for _ in 1:D]
	
	# return expected natural parameters :: Exp_ϕ (for e-step)
	return Exp_ϕ_B(Exp_A, Exp_AᵀA, Exp_C, Exp_R⁻¹, Exp_CᵀR⁻¹C, Exp_R⁻¹C, Exp_CᵀR⁻¹, exp_log_ρ), α_n, γ_n, exp_ρ, exp_log_ρ, Qθ_B(Σ_A, μ_A, Σ_C, μ_C, a_s, b_s)
end

function v_forward_b(ys::Matrix{Float64}, exp_np::Exp_ϕ_B, hpp::HPP_B)
    D, T = size(ys)
    K = size(exp_np.A, 1)

    μs = zeros(K, T)
    Σs = zeros(K, K, T)
	Σs_ = zeros(K, K, T)
	
	Qs = zeros(D, D, T)
	fs = zeros(D, T)

    μ_0 = hpp.μ_0
    Σ_0 = hpp.Σ_0

	# initialise for t = 1
	Σ₀_ = inv(inv(Σ_0) + exp_np.AᵀA)
	Σs_[:, :, 1] = Σ₀_
	
    Σs[:, :, 1] = inv(I + exp_np.CᵀR⁻¹C - exp_np.A*Σ₀_*exp_np.A')
    μs[:, 1] = Σs[:, :, 1]*(exp_np.CᵀR⁻¹*ys[:, 1] + exp_np.A*Σ₀_*inv(Σ_0)μ_0)

	Qs[:, :, 1] = inv(exp_np.R⁻¹ - exp_np.R⁻¹C*Σs[:, :, 1]*exp_np.R⁻¹C')
	fs[:, 1] = Qs[:, :, 1]*exp_np.R⁻¹C*Σs[:, :, 1]*exp_np.A*Σ₀_*inv(Σ_0)*μ_0
		
	# iterate over T
	for t in 2:T
		Σₜ₋₁_ = inv(inv(Σs[:, :, t-1]) + exp_np.AᵀA)
		Σs_[:, :, t] = Σₜ₋₁_
		
		Σs[:, :, t] = inv(I + exp_np.CᵀR⁻¹C - exp_np.A*Σₜ₋₁_*exp_np.A')
    	μs[:, t] = Σs[:, :, t]*(exp_np.CᵀR⁻¹*ys[:, t] + exp_np.A*Σₜ₋₁_*inv(Σs[:, :, t-1])μs[:, t-1])

		Qs[:, :, t] = inv(exp_np.R⁻¹ - exp_np.R⁻¹C*Σs[:, :, t]*exp_np.R⁻¹C')
		fs[:, t] = Qs[:, :, t]*exp_np.R⁻¹C*Σs[:, :, t]*exp_np.A*Σₜ₋₁_*inv(Σs[:, :, t-1])μs[:, t-1]
	end

	return μs, Σs, Σs_, fs, Qs
end

function v_backward(ys::Matrix{Float64}, exp_np::Exp_ϕ_B)
    _, T = size(ys)
    K = size(exp_np.A, 1)

	ηs = zeros(K, T)
    Ψs = zeros(K, K, T)

    # Initialize the filter, t=T, β(x_T-1)
	Ψs[:, :, T] = zeros(K, K)
    ηs[:, T] = ones(K)
	
	Ψₜ = inv(I + exp_np.CᵀR⁻¹C)
	Ψs[:, :, T-1] = inv(exp_np.AᵀA - exp_np.A'*Ψₜ*exp_np.A)
	ηs[:, T-1] = Ψs[:, :, T-1]*exp_np.A'*Ψₜ*exp_np.CᵀR⁻¹*ys[:, T]
	
	for t in T-2:-1:1
		Ψₜ₊₁ = inv(I + exp_np.CᵀR⁻¹C + inv(Ψs[:, :, t+1]))
		
		Ψs[:, :, t] = inv(exp_np.AᵀA - exp_np.A'*Ψₜ₊₁*exp_np.A)
		ηs[:, t] = Ψs[:, :, t]*exp_np.A'*Ψₜ₊₁*(exp_np.CᵀR⁻¹*ys[:, t+1] + inv(Ψs[:, :, t+1])ηs[:, t+1])
	end

	# for t = 1, this correspond to β(x_0), the probability of all the data given the setting of the auxiliary x_0 hidden state.
	Ψ₁ = inv(I + exp_np.CᵀR⁻¹C + inv(Ψs[:, :, 1]))
		
	Ψ_0 = inv(exp_np.AᵀA - exp_np.A'*Ψ₁*exp_np.A)
	η_0 = Ψs[:, :, 1]*exp_np.A'*Ψ₁*(exp_np.CᵀR⁻¹*ys[:, 1] + inv(Ψs[:, :, 1])ηs[:, 1])
	
	return ηs, Ψs, η_0, Ψ_0
end

function parallel_smoother(μs, Σs, ηs, Ψs, η_0, Ψ_0, μ_0, Σ_0)
	K, T = size(μs)
	Υs = zeros(K, K, T)
	ωs = zeros(K, T)

	Υs[:, :, T] = Σs[:, :, T]
	ωs[:, T] = μs[:, T]
	
	for t in 1:(T-1)
		Υs[:, :, t] = inv(inv(Σs[:, :, t]) + inv(Ψs[:, :, t]))
		ωs[:, t] = Υs[:, :, t]*(inv(Σs[:, :, t])μs[:, t] + inv(Ψs[:, :, t])ηs[:, t])
	end

	Υ_0 = inv(inv(Σ_0) + inv(Ψ_0))
	ω_0 = Υ_0*(inv(Σ_0)μ_0 + inv(Ψ_0)η_0)
	
	return ωs, Υs, ω_0, Υ_0
end

function v_pairwise_x(Σs_, exp_np::Exp_ϕ_B, Ψs)
	T = size(Σs_, 3)
    K = size(exp_np.A, 1)

	# cross-covariance is then computed for all time steps t = 0, ..., T−1
	Υ_ₜ₋ₜ₊₁ = zeros(K, K, T)
	
	for t in 1:T-2
		Υ_ₜ₋ₜ₊₁[:, :, t+1] = Σs_[:, :, t+1]*exp_np.A'*inv(I + exp_np.CᵀR⁻¹C + inv(Ψs[:, :, t+1]) - exp_np.A*Σs_[:, :, t+1]*exp_np.A')
	end

	# t=0, the cross-covariance between the zeroth and first hidden states.
	Υ_ₜ₋ₜ₊₁[:, :, 1] = Σs_[:, :, 1]*exp_np.A'*inv(I + exp_np.CᵀR⁻¹C + inv(Ψs[:, :, 1]) - exp_np.A*Σs_[:, :, 1]*exp_np.A')

	# t=T-1, Ψs[T] = 0 special case
	Υ_ₜ₋ₜ₊₁[:, :, T] = Σs_[:, :, T]*exp_np.A'*inv(I + exp_np.CᵀR⁻¹C - exp_np.A*Σs_[:, :, T]*exp_np.A')
	
	return Υ_ₜ₋ₜ₊₁
end

function vb_e_b(ys::Matrix{Float64}, exp_np::Exp_ϕ_B, hpp::HPP_B, smooth_out=false)
    _, T = size(ys)
	# forward pass α_t(x_t)
	μs, Σs, Σs_, fs, Qs = v_forward_b(ys, exp_np, hpp)

	# backward pass β_t(x_t)
	ηs, Ψs, η₀, Ψ₀ = v_backward(ys, exp_np)

	# marginal (smoothed) means, covs, and pairwise beliefs 
	ωs, Υs, ω_0, Υ_0 = parallel_smoother(μs, Σs, ηs, Ψs, η₀, Ψ₀, hpp.μ_0, hpp.Σ_0)

	Υ_ₜ₋ₜ₊₁ = v_pairwise_x(Σs_, exp_np, Ψs)
	
	# hidden state sufficient stats 
	W_A = sum(Υs[:, :, t-1] + ωs[:, t-1] * ωs[:, t-1]' for t in 2:T)
	W_A += Υ_0 + ω_0*ω_0'

	S_A = sum(Υ_ₜ₋ₜ₊₁[:, :, t] + ωs[:, t-1] * ωs[:, t]' for t in 2:T)
	S_A += Υ_ₜ₋ₜ₊₁[:, :, 1] + ω_0*ωs[:, 1]'
	
	W_C = sum(Υs[:, :, t] + ωs[:, t] * ωs[:, t]' for t in 1:T)
	S_C = sum(ωs[:, t] * ys[:, t]' for t in 1:T)

	if (smooth_out) # return variational smoothed mean, cov of xs, ys after completing VBEM iterations
		return ωs, Υs, fs, Qs
	end

	# compute log partition ln Z' (ELBO and convergence check)
	log_Z_ = log_Z(ys, μs, Σs, Σs_, exp_np, hpp)
	
	return HSS(W_C, W_A, S_C, S_A), ω_0, Υ_0, log_Z_
end

function vb_dlm_c(ys::Matrix{Float64}, hpp::HPP_B, hpp_learn=false, max_iter=500, tol=1e-4; init="mle", debug=false)
	D, _ = size(ys)
	K = length(hpp.α)

    hss = missing
    exp_np = missing
	#elbo_prev = -Inf
	el_s = zeros(max_iter)

    if init == "mle"	
		println("\t--- VB using MLE init ---")
		M_mle = MultivariateStats.fit(PPCA, ys; maxoutdim=K)
		σ²_init = M_mle.σ² .* (1 + randn() * 0.2) 
		e_C = M_mle.W[:, 1:K] * (1 + randn() * 0.2)
        R = diagm(ones(D) .* σ²_init)		
        e_R⁻¹ = inv(R)
		e_CᵀR⁻¹C = e_C'*e_R⁻¹*e_C
		e_R⁻¹C = e_R⁻¹*e_C
		e_CᵀR⁻¹ = e_C'*e_R⁻¹
		e_log_ρ = log.(1 ./ diag(R))

        A = [0.0 1.0; 1.0 0.0]
		exp_np = Exp_ϕ_B(A, A'A, e_C, e_R⁻¹, e_CᵀR⁻¹C, e_R⁻¹C, e_CᵀR⁻¹, e_log_ρ)
		#println(exp_np)
        
		hss, _ = vb_e_b(ys, exp_np, hpp)
		if debug
			println("--- Init VB_DLM with MLE ---")
            println("W_A: ", size(hss.W_A))
            println("S_A: ", size(hss.S_A))
            println("W_C: ", size(hss.W_C))
            println("S_C: ", size(hss.S_C))
		end
	end

	# cf. Beal Algorithm 5.3
	for i in 1:max_iter
        #println(i)
		exp_np, α_n, γ_n, exp_ρ, exp_log_ρ, qθ = vb_m_b(ys, hpp, hss)
		hss, ω_0, Υ_0, log_Z_ = vb_e_b(ys, exp_np, hpp)

		# Convergence check
		kl_A_ = sum([kl_A(zeros(K), Diagonal(hpp.α), (qθ.μ_A)[j], qθ.Σ_A) for j in 1:K])
		kl_ρ_ = sum([kl_ρ(hpp.a, hpp.b, qθ.a_s, qθ.b_s) for _ in 1:D])
		kl_C_ = sum([kl_C(zeros(K), hpp.γ, (qθ.μ_C)[s], qθ.Σ_C, exp_ρ[s]) for s in 1:D])
			
		elbo = log_Z_ - (kl_A_ + kl_ρ_ + kl_C_) 
		el_s[i] = elbo

		# if abs(elbo - elbo_prev) < tol
		# 	println("Stopped at iteration: $i")
		# 	el_s = el_s[1:i]
        #     break
		# end
		
        #elbo_prev = elbo
	end
		
	return exp_np, el_s[end]
end

# adapted from "Stats with Julia"
function compareDigits(train_y, train_labels, M, dA, dB)
    imA = train_y[:, :, findall(x -> x == dA, train_labels)]
    imB = train_y[:, :, findall(x -> x == dB, train_labels)]

    yA = hcat([vcat(float.(imA[:, :, t])...) for t in 1:size(imA, 3)]...)
    yB = hcat([vcat(float.(imB[:, :, t])...) for t in 1:size(imB, 3)]...)

    xA, xB = M'*yA, M'*yB
    scatter(xB[2, :], xB[1, :], c=:red, label="Digit $(dB)", ms=1.5, msw=0,
    legend = :topright, xlabel="PC 1", ylabel="PC 2")
    scatter!(xA[2, :], xA[1, :], c=:blue, label="Digit $(dA)", ms=1.5, msw=0)
end

# compare principal components between different digits and 0
function compare_0(train_y, train_labels, number, M)
    img_0 = train_y[:, :, findall(x -> x == 0, train_labels)]
    img_n0 = train_y[:, :, findall(x -> x == number, train_labels)]

    y0 = hcat([vcat(float.(img_0[:, :, t])...) for t in 1:size(img_0, 3)]...)
    yn0 = hcat([vcat(float.(img_n0[:, :, t])...) for t in 1:size(img_n0, 3)]...)

    x0, xn0 = M'*y0, M'*yn0

    p = scatter(x0[1, :], x0[2, :], c=:red, label="Digit 0", ms=1.5, msw=0, 
    legend = :topright, xlabel="PC 1", ylabel="PC 2")
    scatter!(xn0[1, :], xn0[2, :], c=:blue, label="Digit $number", ms=1.5, msw=0)
    return p
end

function plot_number(train_y, train_labels, number, mode="pca")
    # get the image of number = number only and compare with 0
    img_n = train_y[:, :, findall(x -> x == number, train_labels)]
    y_n = hcat([vcat(float.(img_n[:, :, t])...) for t in 1:size(img_n, 3)]...)
    y_n = zscore(y_n, 1)

    img_0 = train_y[:, :, findall(x -> x == 0, train_labels)]
    y_0 = hcat([vcat(float.(img_0[:, :, t])...) for t in 1:size(img_0, 3)]...)
    y_0 = zscore(y_0, 1)

    y_sum = hcat(y_n, y_0)

    x_n = missing
    x_0 = missing
    p = missing

    if mode == "pca"
        pca = MultivariateStats.fit(PCA, y_sum; maxoutdim=2)
        M_pca = projection(pca)
        x_n = M_pca' * y_n
        p = scatter(x_n[2, :], x_n[1, :], c=:red, label="Digit $number", ms=1.5, msw=0, 
        legend = :topright, xlabel="PC 1", ylabel="PC 2")
        x_0 = M_pca' * y_0
        scatter!(x_0[2, :], x_0[1, :], c=:blue, label="Digit 0", ms=1.5, msw=0)
    end

    if mode == "ppca"
        C, _, _ = vb_ppca_k2(y_sum, 2, false, mode="mle")
        M_vb = svd(C).U
        x_n = M_vb' * y_n
        p = scatter(x_n[2, :], x_n[1, :], c=:red, label="Digit $number", ms=1.5, msw=0, 
        legend = :topright, xlabel="PC 1", ylabel="PC 2")
        x_0 = M_vb' * y_0
        scatter!(x_0[2, :], x_0[1, :], c=:blue, label="Digit 0", ms=1.5, msw=0)
    end

    return p
end

function test_pattern(train_y, train_labels, length=100, mode="pca")
    img_1 = train_y[:, :, findall(x -> x == 1, train_labels)]
    y_1 = hcat([vcat(float.(img_1[:, :, t])...) for t in 1:size(img_1, 3)]...)
    y_1 = zscore(y_1, 1)

    img_0 = train_y[:, :, findall(x -> x == 0, train_labels)]
    y_0 = hcat([vcat(float.(img_0[:, :, t])...) for t in 1:size(img_0, 3)]...)
    y_0 = zscore(y_0, 1)

    pattern_length = length
    y_pattern = zeros(784, length)

    for i in 1:pattern_length
        if i % 2 == 0
            y_pattern[:, i] = y_0[:, i]
        else
            y_pattern[:, i] = y_1[:, i]
        end
    end

    if mode == "pca"
        pca = MultivariateStats.fit(PCA, y_pattern; maxoutdim=2)
        x_pca = predict(pca, y_pattern)

        # println(size(x_pca))
        return x_pca
    end

    if mode == "ppca"
        _, _, els, xs_ppca, crsvar_ppca = vb_ppca_k2(y_pattern, 1, false, mode="mle")

        xs_stds = sqrt.(hcat([diag(crsvar_ppca[:, :, t]) for t in 1:pattern_length]...))
        #println(size(xs_stds))

		println("PPCA Final ELBO: ", els[end])
        return xs_ppca, xs_stds[:, 1]
    end

    if mode == "b"
        K = 2
        α = ones(K)
        γ = ones(K)
        a = 2
        b = 0.001
        μ_0 = zeros(K)
        Σ_0 = Matrix{Float64}(I*1e5, K, K)
        hpp = HPP_B(α, γ, a, b, μ_0, Σ_0)
        exp_np, els = vb_dlm_c(y_pattern, hpp, false, 1)
        ωs, Υs = vb_e_b(y_pattern, exp_np, hpp, true)

		println("VB Final ELBO: ", els)
        xs_stds = sqrt.(hcat([diag(Υs[:, :, t]) for t in 1:pattern_length]...))
        return exp_np.A, ωs, xs_stds
    end
end

function test_MNIST(test_prop=100, standardise=true, method="pca")
    train_y, train_labels = MNIST(split=:train)[:]
    T = length(train_labels)
    
    """ Optional:
    Use a proportion of the trainning data to test quicker/visualise better
    """
    test_length = Int.(T * test_prop/100)
    train_y, train_labels = train_y[:, :, 1:test_length], train_labels[1:test_length]
    y = hcat([vcat(Float64.(train_y[:, :, t])...) for t in 1:test_length]...)

    if standardise
        y = zscore(y, 1)
    end

    plots = []
    M = missing

    if method == "pca"
        pca = MultivariateStats.fit(PCA, y; maxoutdim=2)
        M = projection(pca)
    end

    """
    Package PPCA options like :em and :bayes does not run with SingularExcepion
    despite standarisation
    """
    # if method == "em"
    #     M_em = MultivariateStats.fit(PPCA, y; method=(:em), maxoutdim=2)
    #     M = projection(M_em)
    # end

    # if method == "bayes"
    #     M_bay = MultivariateStats.fit(PPCA, y; method=(:bayes), maxoutdim=2)
    #     M = projection(M_bay)
    # end

    if method == "vbem"
        C, _, _, _, _ = vb_ppca_k2(y, 1, false, debug=false)
        M = svd(C).U
    end

    for k in 1:5
        push!(plots, compareDigits(train_y, train_labels, M, 2k-2, 2k-1))
    end

    # for n in 1:9
    #     push!(plots, compare_0(train_y, train_labels, n, M))
    # end

    p = plot(plots..., size = (1600, 1000), margin = 5mm)
    #title!(p, "$method, $test_prop % data")
    display(p)
end

function gen_pca_pattern(train_y, train_labels)
	xs_pca = test_pattern(train_y, train_labels, 1000, "pca")
	xs_pca = abs.(xs_pca[:, 91:110])
	observation_indices = 1:20 
	odd_data = xs_pca[:, 1:2:end]
	even_data = xs_pca[:, 2:2:end]
	odd_indices = observation_indices[1:2:end]
	even_indices = observation_indices[2:2:end]

	p = plot3d(observation_indices, xs_pca[1, :], xs_pca[2, :], 
		xlabel = "Observation Index", ylabel = "PC 1", zlabel = "PC 2",
		title = "", lw = 1.5, ls=:dash, color = :blue, label="")

	scatter!(p, odd_indices, odd_data[1, :], odd_data[2, :], 
			ms=3, color = :red, label="1")

	scatter!(p, even_indices, even_data[1, :], even_data[2, :], 
			ms=3, color = :black, label="0")
	display(p)

	println("PCA Complete")
end

function gen_ppca_pattern(train_y, train_labels)
	xs_ppca, xs_stds = test_pattern(train_y, train_labels, 1000, "ppca")
	xs_ppca = abs.(xs_ppca[:, 91:110])
	observation_indices = 1:20 

	odd_data = xs_ppca[:, 1:2:end]
	even_data = xs_ppca[:, 2:2:end]
	odd_indices = observation_indices[1:2:end]
	even_indices = observation_indices[2:2:end]

	p_ppca = plot3d(observation_indices, xs_ppca[1, :], xs_ppca[2, :], 
	xlabel = "Observation Index", ylabel = "PC 1", zlabel = "PC 2",
	title = "", lw = 1.5, color = :blue, ls =:dash, label="")

	credible_interval = 1.96 .* xs_stds

	for i in 1:20 
		# Error bars along Y-axis
		plot!([observation_indices[i], observation_indices[i]], 
			[xs_ppca[1, i] - credible_interval[1], xs_ppca[1, i] + credible_interval[1]], 
			[xs_ppca[2, i], xs_ppca[2, i]], color=:green, lw = 1, ls=:dash, label="")

		# Error bars along Z-axis
		plot!([observation_indices[i], observation_indices[i]], 
			[xs_ppca[1, i], xs_ppca[1, i]], 
			[xs_ppca[2, i] - credible_interval[2], xs_ppca[2, i] + credible_interval[2]], 
			color=:green, lw=1, ls=:dash, label="")
	end

	scatter!(p_ppca, odd_indices, odd_data[1, :], odd_data[2, :], 
			ms = 3, color = :red, label="1")

	scatter!(p_ppca, even_indices, even_data[1, :], even_data[2, :], 
			ms = 3, color = :black, label="0")

	display(p_ppca)
	println("PPCA Complete")
end

function gen_vb_pattern(train_y, train_labels)
	_, xs, xs_stds = test_pattern(train_y, train_labels, 1000, "b")

	# println("1 ", xs_stds[:, 1])
	# println("End ", xs_stds[:, end])
	# println("100 ", xs_stds[:, 100])
	xs = abs.(xs[:, 91:110])
	xs_stds = xs_stds[:, 91:110]
	observation_indices = 1:20 

	p = plot3d(observation_indices, xs[1, :], xs[2, :], 
		xlabel = "Observation Index", ylabel = "PC 1", zlabel = "PC 2",
		title = "", lw = 1.5, ls=:dash, color = :blue, label="")

	odd_data = xs[:, 1:2:end]
	even_data = xs[:, 2:2:end]
	odd_indices = observation_indices[1:2:end]
	even_indices = observation_indices[2:2:end]

	credible_interval = 1.96 .* xs_stds
	
	for i in 1:20 
		# Error bars along Y-axis
		plot!([observation_indices[i], observation_indices[i]], 
			[xs[1, i] - credible_interval[1, i], xs[1, i] + credible_interval[1, i]], 
			[xs[2, i], xs[2, i]], color=:green, lw = 1, ls=:dash, label="")

		# Error bars along Z-axis
		plot!([observation_indices[i], observation_indices[i]], 
			[xs[1, i], xs[1, i]], 
			[xs[2, i] - credible_interval[2, i], xs[2, i] + credible_interval[2, i]], 
			color=:green, lw=1, ls=:dash, label="")
	end

	scatter!(p, odd_indices, odd_data[1, :], odd_data[2, :], 
	ms = 3, color = :red, label="1")

	scatter!(p, even_indices, even_data[1, :], even_data[2, :], 
	ms = 3, color = :black, label="0")
	display(p)
	println("VB Complete")
end

function gen_MNIST_O_similar(train_y, train_labels)
	plots_pca = []
	for k in [1, 6, 8, 9]
		push!(plots_pca, plot_number(train_y, train_labels, k, "pca"))
	end
	p_pca = plot(plots_pca..., size = (1600, 1000), margin = 5mm)
	display(p_pca)

	plots_ppca = []
	for k in [1, 6, 8, 9]
		push!(plots_ppca, plot_number(train_y, train_labels, k, "ppca"))
	end
	p_ppca = plot(plots_ppca..., size = (1600, 1000), margin = 5mm)
	display(p_ppca)

end

function time_series_ppca_var()
	_, _, xs_stds = test_pattern(train_y, train_labels, 500, "b")

	p = plot(xs_stds[1, :], title="Time series PPCA", label="", ylabel="σ", xlabel="T")
	#ylims!(p, 0.05, 0.08)
	display(p)

	p_2 = plot(xs_stds[2, :], title="Time series PPCA", label="", ylabel="σ", xlabel="T")
	#ylims!(p_2, 0.05, 0.12)
	display(p_2)
end

function ppca_var()
	_, xs_stds = test_pattern(train_y, train_labels, 500, "ppca")

	p = plot(ones(500) .* xs_stds[1], title="PPCA", label="", ylabel="σ", xlabel="T")
	ylims!(p, 0.0, 0.08)
	display(p)

	p_2 = plot(ones(500) .* xs_stds[2], title="PPCA", label="", ylabel="σ", xlabel="T")
	ylims!(p_2, 0.0, 0.12)
	display(p_2)

end

"""
Collection of tests, uncomment to run
"""
#test_MNIST(100, true)
#test_MNIST(100, true, "vbem")

train_y, train_labels = MNIST(split=:train)[:]
train_y, train_labels = train_y[:, :, 1:12000], train_labels[1:12000]

time_series_ppca_var()
ppca_var()

#gen_pca_pattern(train_y, train_labels)

#gen_ppca_pattern(train_y, train_labels)

#gen_vb_pattern(train_y, train_labels)
