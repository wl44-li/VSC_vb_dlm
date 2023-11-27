begin
	using Turing
	using Distributions
	using LinearAlgebra
	using MCMCChains
	using Random
	using StatsPlots
    using DataFrames
end

function gen_data(A, C, Q, R, μ_0, T)
	# re-producibility
	Random.seed!(10)

	K = size(A, 1)
	D = size(C, 1)
	x = zeros(K, T)
	y = zeros(D, T)

	for t in 1:T
		x[:, t] = rand(MvNormal(zeros(K), Q))
		y[:, t] = C * x[:, t] + rand(MvNormal(zeros(D), R)) 
	end

	return y, x
end

"""
Primiary testing shows poor convergence results in 3 hours, T=500, trivial test
"""
@model function ppca_Turing(y, K, a, b, γ)
    P, T = size(y)

    # Validate that P > 1 for PPCA to be meaningful
    @assert P > 1 "PPCA requires P > 1 (for each observation instance)."

    # Priors
    τ ~ Gamma(a, b)
    R = diagm(ones(P) .* (1/τ))

    # Priors for the loading matrix C using a multivariate normal distribution
    C ~ filldist(MvNormal(zeros(K), 1/(τ * γ) * I), P)

	#println("C ", size(C))
    # Priors for the latent variables x using a multivariate normal distribution
    x ~ filldist(MvNormal(zeros(K), I), T)
	#println("x ", size(x))

    for t in 1:T
        y[:, t] ~ MvNormal(C'*x[:, t], R)
    end
end


@model function ppca_alt(y, K)
    P, T = size(y)

    # Validate that P > 1 for PPCA to be meaningful
    @assert P > 1 "PPCA requires P > 1 (for each observation instance)."

    # Priors
    τ ~ Gamma(1.1, 20)
    R = diagm(ones(P) .* (1/τ))

    # Priors for the loading matrix C using a multivariate normal distribution
    C ~ filldist(MvNormal(zeros(K), I), P)

    x ~ filldist(MvNormal(zeros(K), I), T)
	#println("x ", size(x))

    for t in 1:T
        y[:, t] ~ MvNormal(C'*x[:, t], R)
    end
end

function nuts_ppca(y, mcmc=5000, burn_in=1000)
    model = ppca_Turing(y, 1, 2, 1e-3, 1)
    chain = Turing.sample(model, NUTS(), mcmc)
    return chain[burn_in+1:end]
end

function nuts_ppca_alt(y, mcmc=5000, burn_in=1000)
    model = ppca_alt(y, 1)
    chain = Turing.sample(model, NUTS(), mcmc)
    return chain[burn_in+1:end]
end

function hmc_ll(y, mcmc=5000, burn_in=1000)
    model = ppca_Turing(y, 1, 2, 1e-3, 1)
    chain = Turing.sample(model, HMC(0.05, 5), mcmc)
    return chain[burn_in+1:end]
end

T = 500
C_d2k1 = reshape([1.0, 0.5], 2, 1)
R_2 = Diagonal([1.0, 1.0])
y, x = gen_data([0.0], C_d2k1, [1.0], R_2, 0.0, T)

#println(size(y), size(x))

nuts_chain = nuts_ppca_alt(y)

