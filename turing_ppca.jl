begin
	using Turing
	using Distributions
	using LinearAlgebra
	using MCMCChains
	using Random
	using StatsPlots
    using DataFrames
end

function gen_data(A, C, Q, R, μ_0, Σ_0, T)
	Random.seed!(10)
	K, _ = size(A)
	D, _ = size(C)
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

@model function ppca_Turing(y, K, a, b, γ)
    P, T = size(y)

    # Validate that P > 1 for PPCA to be meaningful
    @assert P > 1 "PPCA requires P > 1 (for each observation instance)."

    # Priors
    τ ~ Gamma(a, b)
    R = diagm(ones(P) .* (1/τ))

    # Priors for the loading matrix C using a multivariate normal distribution
    C ~ filldist(MvNormal(zeros(K), 1/(τ * γ) * I), P)

    # Priors for the latent variables x using a multivariate normal distribution
    x ~ filldist(MvNormal(zeros(K), I), T)

    for t in 1:T
        y[:, t] ~ MvNormal(C*x[:, t], R)
    end
end