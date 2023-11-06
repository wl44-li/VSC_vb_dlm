# import Pkg
# Pkg.add("Turing")

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
end

@model function ll_Turing(y)
    T = length(y)
    
    # Priors
    r ~ InverseGamma(0.1, 0.1)
    q ~ InverseGamma(0.1, 0.1)
    
    x = Vector(undef, T)
    x[1] ~ Normal(0.0, 1.0)
    y[1] ~ Normal(x[1], sqrt(r))
    
    for t in 2:T
        # State transition
        x[t] ~ Normal(x[t-1], sqrt(q))
        y[t] ~ Normal(x[t], sqrt(r))
    end
end

function nuts_ll(y, mcmc=4000, burn_in=1000)
    model = ll_Turing(y)
    chain = Turing.sample(model, NUTS(), mcmc)
    return chain[burn_in+1:end]
end

function hmc_ll(y, mcmc=4000, burn_in=1000)
    model = ll_Turing(y)
    chain = Turing.sample(model, HMC(0.05, 5), mcmc)
    return chain[burn_in+1:end]
end

function test_nuts_x(sd, max_T = 100)
    Random.seed!(sd)
    Q = 1.0
    R = 1.0
    println("Ground-truth r = $R, q = $Q")
    y, x_true = gen_data(1.0, 1.0, Q, R, 0.0, max_T)

    nuts_chain = nuts_ll(y)
    # rs, qs = nuts_chain[:r], nuts_chain[:q]
    # p_r, p_q = plot(rs, label = "NUTS r"), plot(qs, label = "NUTS q")
    #display(p_r)
    #display(p_q)

    n = 200
    subset_chain = nuts_chain[1:n]
    x_samples = [subset_chain[Symbol("x[$t]")] for t in 1:max_T]
    x_matrix = hcat([Array(x_samples[t])[:] for t in 1:max_T]...)
    p_xs = plot(x_matrix', legend=false, title="NUTS Trace Plot of x[0:T] ($n iterations)", xlabel="x[0:T]", ylabel="")
    display(p_xs)

end

function test_hmc_x(sd, max_T)
    Random.seed!(sd)
    Q = 1.0
    R = 1.0
    println("Ground-truth r = $R, q = $Q")
    y, x_true = gen_data(1.0, 1.0, Q, R, 0.0, max_T)
    hmc_chain = hmc_ll(y)
    n = 200
    hmc_subset_chain = hmc_chain[1:n]
    x_ss = [hmc_subset_chain[Symbol("x[$t]")] for t in 1:max_T]
    x_ma = hcat([Array(x_ss[t])[:] for t in 1:max_T]...)
    p_xss = plot(x_ma', legend=false, title="HMC Trace Plot of x[0:T] ($n iterations)", xlabel="x[0:T]", ylabel="")
    display(p_xss)
end

#test_nuts_x(111, 50)

#test_hmc_x(111, 50)

function test_nuts(y)
    T = length(y)
    nuts_chain = nuts_ll(y)
    rs, qs = nuts_chain[:r], nuts_chain[:q]

    x_means = Vector{Float64}(undef, T)
    x_vars = Vector{Float64}(undef, T)

    for t in 1:T
        # Extract the samples for x[t]
        samples = nuts_chain[Symbol("x[$t]")].data
        x_means[t] = mean(samples)
        x_vars[t] = var(samples)
    end

    return x_means, x_vars, rs, qs
end

function test_hmc(y)
    T = length(y)
    hmc_chain = hmc_ll(y)
    rs, qs = hmc_chain[:r], hmc_chain[:q]

    x_means = Vector{Float64}(undef, T)
    x_vars = Vector{Float64}(undef, T)

    for t in 1:T
        # Extract the samples for x[t]
        samples = hmc_chain[Symbol("x[$t]")].data
        x_means[t] = mean(samples)
        x_vars[t] = var(samples)
    end

    return x_means, x_vars, rs, qs
end

#main()