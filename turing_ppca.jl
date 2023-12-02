begin
	using Turing
	using Distributions
	using LinearAlgebra
	using MCMCChains
	using Random
	using StatsPlots
    using DataFrames
end

function gen_data_ppca(A, C, Q, R, T)
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
Corrected model, NUTS (~ 1 hr), HMC (~ 20 mins), iteration 5000
NUTS (~ ), HMC (~ 10 mins), iteration 3000
"""
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

# Turing tutorial example
@model function pPCA_ARD(Y)
    # Dimensionality of the problem. Y needs to be transposed 
    T, P = size(Y)

    # latent variable X
    X ~ filldist(Normal(), P, T)

    # weights/loadings w with Automatic Relevance Determination part
    α ~ filldist(Gamma(1.0, 1.0), P)
    W ~ filldist(MvNormal(zeros(P), 1.0 ./ sqrt.(α)), P)

    mu = (W' * X)'

    τ ~ Gamma(1.0, 1.0)
    return Y ~ arraydist([MvNormal(m, 1.0 / sqrt(τ)) for m in eachcol(mu)])
end

function nuts_ppca_alt(y, K, mcmc=3000, burn_in=1000)
    model = ppca_alt(y, K)
    chain = Turing.sample(model, NUTS(), mcmc)
    return chain[burn_in+1:end]
end

function hmc_ppca(y, K, mcmc=3000, burn_in=1000)
    model = ppca_alt(y, K)
    chain = Turing.sample(model, HMC(0.05, 5), mcmc)
    return chain[burn_in+1:end]
end

function test_nuts()
    T = 500
    C_d2k1 = reshape([1.0, 0.5], 2, 1)
    R_2 = Diagonal([1.0, 1.0])
    y, _ = gen_data_ppca([0.0], C_d2k1, [1.0], R_2, T)

    nuts_chain = nuts_ppca_alt(y, 1)

    τs = nuts_chain[:τ]
    p_τ = plot(τs, label = "NUTS τ")
    display(p_τ)
    p_t = density(τs, label = "τ")
    display(p_t)

    c1s, c2s = nuts_chain[Symbol("C[1,1]")].data, nuts_chain[Symbol("C[1,2]")].data
    
    p1 = density(c1s, label = "C[1, 1]")
    display(p1)
    p2 = density(c2s, label = "C[2, 1]")
    display(p2)

    return nuts_chain
end

function test_hmc()
    T = 500
    C_d2k1 = reshape([1.0, 0.5], 2, 1)
    R_2 = Diagonal([1.0, 1.0])
    y, _ = gen_data_ppca([0.0], C_d2k1, [1.0], R_2, T)

    hmc_chain = hmc_ppca(y, 1)

    τs = hmc_chain[:τ]
    p_τ = plot(τs, label = "HMC τ")
    display(p_τ)
    p_t = density(τs, label = "τ")
    display(p_t)

    c1s, c2s = hmc_chain[Symbol("C[1,1]")].data, hmc_chain[Symbol("C[1,2]")].data

    p1 = density(c1s, label = "C[1, 1]")
    display(p1)
    p2 = density(c2s, label = "C[2, 1]")
    display(p2)
    return hmc_chain
end

#hmc_chain_1 = test_hmc()

#nuts_chain_1 = test_nuts()

# """
# Test with K = 2, P = 3? and compare with VB PPCA
# """

# function test_hmc_p3()
#     T = 500
# 	C_d3k2 = [1.0 0.0; 0.2 1.0; 0.9 0.1] 
#     R_3 = Diagonal([1.0, 1.0, 1.0])
#     y, _ = gen_data(zeros(2, 2), C_d3k2, Diagonal([1.0, 1.0]), R_3, zeros(2), T)

#     hmc_chain = hmc_ppca(y, 2)

#     τs = hmc_chain[:τ]
#     p_τ = plot(τs, label = "HMC τ")
#     display(p_τ)
#     p_t = density(τs, label = "τ")
#     display(p_t)

#     # c1s, c2s = hmc_chain[Symbol("C[1,1]")].data, hmc_chain[Symbol("C[1,2]")].data

#     # p1 = density(c1s, label = "C[1, 1]")
#     # display(p1)
#     # p2 = density(c2s, label = "C[2, 1]")
#     # display(p2)
#     return hmc_chain
# end

# hmc_chain_2 = test_hmc_p3()