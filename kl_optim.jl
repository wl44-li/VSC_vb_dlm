import Pkg
Pkg.add("MCMCChains")
Pkg.add("StatsBase")
Pkg.add("PDMats")
Pkg.add("Statistics")
Pkg.add("StatsPlots")
Pkg.add("DataFrames")
Pkg.add("SpecialFunctions")
Pkg.add("LinearAlgebra")
Pkg.add("Plots")
Pkg.add("Random")
Pkg.add("StateSpaceModels")
Pkg.add("Distributions")
Pkg.add("MultivariateStats")
Pkg.add("StatsFuns")
using SpecialFunctions
using LinearAlgebra

function kl_Wishart(ν_q, S_q, ν_0, S_0)
	k = size(S_0, 1)
	# Beale thesis
	term1 = 0.5*(ν_0 - ν_q)*k*log(2) + 0.5*ν_0*logdet(S_0) - 0.5*ν_q*logdet(S_q) + sum(loggamma((ν_0 + 1 - i)/2.0) for i in 1:k) - sum(loggamma((ν_q + 1 - i)/2.0) for i in 1:k)
	
    term2 = (ν_q - ν_0) * (sum(digamma((ν_q + 1 - i)/2.0) for i in 1:k) + k*log(2) + logdet(S_q))
	
    term3 = ν_q * tr(inv(S_0) * S_q - I)
    return term1 + 0.5 * (term2 + term3) 
end


function kl_gamma(a_0, b_0, a_s, b_s)
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


struct HPP_D
    α::Float64
    β::Float64 
	a::Float64 
    b::Float64 
    μ_0::Vector{Float64} # auxiliary hidden state mean
    Σ_0::Matrix{Float64} # auxiliary hidden state co-variance
end


struct Q_Gamma
	a
	b
	α
	β
end


function update_hyp_D(hpp::HPP_D, Q_gam::Q_Gamma)
	b_s = Q_gam.b
	D = length(b_s)
	a_s = Q_gam.a * ones(D)
	exp_ρ = a_s ./ b_s 
	exp_log_ρ = [(digamma(Q_gam.a) - log(b_s[i])) for i in 1:D]
    d = mean(exp_ρ)
    c = mean(exp_log_ρ)
    
    # Update using fixed point equations
	a = hpp.a		
	α = hpp.α
    for _ in 1:100
        ψ_a = digamma(a)
        ψ_a_p = trigamma(a)
        
        a_new = a * exp(-(ψ_a - log(a) + log(d) - c) / (a * ψ_a_p - 1))
		a = a_new

		# check convergence
        if abs(a_new - a) < 1e-5
            break
        end
    end
    
    # Update `b` using the converged value of `a`
    b = a/d

	β_s = Q_gam.β
	K = length(β_s)
	α_s = Q_gam.α * ones(K)
	exp_𝛐 = α_s ./ β_s 
	exp_log_𝛐 = [(digamma(Q_gam.α) - log(β_s[i])) for i in 1:K]
    d_ = mean(exp_𝛐)
    c_ = mean(exp_log_𝛐)

	for _ in 1:100
        ψ_α = digamma(α)
        ψ_α_p = trigamma(α)
        
        α_new = α * exp(-(ψ_α - log(α) + log(d_) - c_) / (α * ψ_α_p - 1))
		α = α_new

		# check convergence
        if abs(α_new - α) < 1e-5
            break
        end
    end
	β = α/d_
	
	return a, b, α, β
end


struct HSS
	W_C::Array{Float64, 2}
	W_A::Array{Float64, 2}
	S_C::Array{Float64, 2}
	S_A::Array{Float64, 2}
end


function gen_data(A, C, Q, R, μ_0, Σ_0, T)

    if length(A) == 1 && length(C) == 1 # uni-variate
		x = zeros(T)
		y = zeros(T)
		
		for t in 1:T
		    if t == 1
		        x[t] = μ_0 + sqrt(Q) * randn()
		    else
		        x[t] = A * x[t-1] + sqrt(Q) * randn()
		    end
		    	y[t] = C * x[t] + sqrt(R) * randn()
		end
		return y, x

    else
        K, _ = size(A)
        D, _ = size(C)
        x = zeros(K, T)
        y = zeros(D, T)

        x[:, 1] = rand(MvNormal(A*μ_0, A'*Σ_0*A + Q))
        y[:, 1] = C * x[:, 1] + rand(MvNormal(zeros(D), R))

        for t in 2:T
            if (tr(Q) != 0)
                x[:, t] = A * x[:, t-1] + rand(MvNormal(zeros(K), Q))
            else
                x[:, t] = A * x[:, t-1] # Q zero matrix special case of PPCA
            end

            if D == 1
                y[:, t] = C * x[:, t] + rand(MvNormal(zeros(D), sqrt.(R))) # linear growth 
            else
                y[:, t] = C * x[:, t] + rand(MvNormal(zeros(D), R)) 
            end
        end

	    return y, x
    end
end
