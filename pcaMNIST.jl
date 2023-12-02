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
pyplot()

""" DEBUG zero-forcing VB-PPCA
Consider reduce EM iterations (early stop before iter 100, 50, 30, 10)

Consider standardise data? (get rid of the 0s from the white pixels)
"""

function compareDigits(train_y, train_labels, M, dA, dB)
    imA = train_y[:, :, findall(x -> x == dA, train_labels)]
    imB = train_y[:, :, findall(x -> x == dB, train_labels)]

    yA = hcat([vcat(float.(imA[:, :, t])...) for t in 1:size(imA, 3)]...)
    yB = hcat([vcat(float.(imB[:, :, t])...) for t in 1:size(imB, 3)]...)

    xA, xB = M'*yA, M'*yB
    # default(ms=0.8, msw=0, xlims=(-5, 12.5), ylims=(-7.5, 7.5),
    #         legend = :topright, xlabel="PC 1", ylabel="PC 2")
    scatter(xA[1, :], xA[2, :], c=:red, label="Digit $(dA)", ms=0.8, msw=0, xlims=(-5, 12.5), ylims=(-7.5, 7.5),
    legend = :topright, xlabel="PC 1", ylabel="PC 2")
    scatter!(xB[1, :], xB[2, :], c=:blue, label="Digit $(dB)", ms=0.8, msw=0)
end

function compare_0(train_y, train_labels, M)
    img_0 = train_y[:, :, findall(x -> x == 0, train_labels)]
    img_n0 = train_y[:, :, findall(x -> x != 0, train_labels)]

    y0 = hcat([vcat(float.(img_0[:, :, t])...) for t in 1:size(img_0, 3)]...)
    yn0 = hcat([vcat(float.(img_n0[:, :, t])...) for t in 1:size(img_n0, 3)]...)

    x0, xn0 = M'*y0, M'*yn0

    scatter(x0[1, :], x0[2, :], c=:red, label="Digit 0", ms=0.8, msw=0, xlims=(-5, 12.5), ylims=(-7.5, 7.5),
    legend = :topright, xlabel="PC 1", ylabel="PC 2")
    scatter!(xn0[1, :], xn0[2, :], c=:blue, label="Digit {1:9}", ms=0.8, msw=0)
end

function plot_number(train_y, train_labels, number, M)
    # get the image of number = number only
    img_n = train_y[:, :, findall(x -> x == number, train_labels)]
    y_n = hcat([vcat(float.(img_n[:, :, t])...) for t in 1:size(img_n, 3)]...)

    x_n = M' * y_n
    scatter(x_n[1, :], x_n[2, :], c=:red, label="Digit $number", ms=0.8, msw=0, xlims=(-5, 12.5), ylims=(-7.5, 7.5),
    legend = :topright, xlabel="PC 1", ylabel="PC 2")
end

function test_MNIST(test_prop=100, standardise = true, method = "pca")
    train_y, train_labels = MNIST(split=:train)[:]

    T = length(train_labels)
    
    """ Optional:
    -- take a proportion of the trainning data to test quicker
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

    if method == "mle"
        mle = MultivariateStats.fit(PPCA, y; maxoutdim=2)
        M = projection(mle)
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
        C = vb_ppca_k2(y, 100, false, debug=false)
        M = svd(C).U
    end

    for k in 1:5
        push!(plots, compareDigits(train_y, train_labels, M, 2k-2,2k-1))
    end

    p = plot(plots..., size = (1600, 1000), margin = 5mm)
    title!(p, "$method, $test_prop % data")
    display(p)
end

train_y, train_labels = MNIST(split=:train)[:]
train_y, train_labels = train_y[:, :, 1:30000], train_labels[1:30000]

T = size(train_y, 3)
y = hcat([vcat(Float64.(train_y[:, :, t])...) for t in 1:T]...)

#println("dimension of Y(obs): ", size(y))

# 1s
img_1 = train_y[:, :, findall(x -> x == 1, train_labels)]
y_1s = hcat([vcat(float.(img_1[:, :, t])...) for t in 1:size(img_1, 3)]...)

#println("dimension of 1s (obs): ", size(y_1s))

# pca = MultivariateStats.fit(PCA, y_1s; maxoutdim=2)
# M_pca = projection(pca)
# x_1s = M_pca' * y_1s
# scatter(x_1s[1, :], x_1s[2, :], c=:red, label="Digit 1 (PCA)", ms=0.8, msw=0, xlims=(-7.5, 7.5), ylims=(-7.5, 7.5),
# legend = :topright, xlabel="PC 1", ylabel="PC 2")


# mle = MultivariateStats.fit(PPCA, y_1s; maxoutdim=2)
# M_mle = projection(mle)
# x_1s_mle = M_mle' * y_1s
# scatter(x_1s_mle[1, :], x_1s_mle[2, :], c=:orange, label="Digit 1 (MLE)", ms=0.8, msw=0, xlims=(-7.5, 7.5), ylims=(-7.5, 7.5),
# legend = :topright, xlabel="PC 1", ylabel="PC 2")


# M_mle_k2 = svd(mle.W[:, 1:2]).U
# x_1s_k2 = M_mle_k2' * y_1s
# scatter(x_1s_k2[1, :], x_1s_k2[2, :], c=:blue, label="Digit 1 (MLE, K2)", ms=0.8, msw=0, xlims=(-7.5, 7.5), ylims=(-7.5, 7.5),
# legend = :topright, xlabel="PC 1", ylabel="PC 2")

y_st = zscore(y_1s, 1)
C, σ², els = vb_ppca_k2(y_st, 2000, false)
M_vb = svd(C).U
x_1s_vb2 = M_vb' * y_st
scatter(x_1s_vb2[2, :], x_1s_vb2[1, :], c=:black, label="Digit 1 (VB-std)", ms=1.5, msw=0, 
legend = :topleft, xlabel="PC 1", ylabel="PC 2")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           

function test_vb_ppca(y, n=10)
    plots = []
    elss = []
	p_elbo = plot()

    for rep in 1:n

        C, els = missing, missing
        if rep%2 == 0
            C, _, els = vb_ppca_k2(y, 2000, false, mode = "mle")

        else
            C, _, els = vb_ppca_k2(y, 2000, false, mode = "random")
        end
        M_vb = svd(C).U
        x_1s_vb2 = M_vb' * y_st

        push!(plots, scatter(x_1s_vb2[2, :], x_1s_vb2[1, :], c=:black, label="Digit 1 (VB-std)", ms=1.5, msw=0, 
        legend = :topleft, xlabel="PC 1", ylabel="PC 2"))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        plot!(p_elbo, els, label="", ylabel="ELBO", xlabel="Iterations")	
        push!(elss, els)
        println("ELBOs end: ", els[end])
    end

    display(p_elbo)

    return plots, elss
end

plot_1s, elss = test_vb_ppca(y_st, 20)

# mle_std = MultivariateStats.fit(PPCA, y_st; maxoutdim=2)
# M_mle_k2 = svd(mle_std.W[:, 1:2]).U
# x_1s_k2 = M_mle_k2' * y_st
# scatter(x_1s_k2[1, :], x_1s_k2[2, :].*(-1), c=:blue, label="Digit 1 (MLE, K2 std)", ms=1.5, msw=0,
# legend = :topleft, xlabel="PC 1", ylabel="PC 2")


# Os
# img_0 = train_y[:, :, findall(x -> x == 0, train_labels)]
# y_0s = hcat([vcat(float.(img_0[:, :, t])...) for t in 1:size(img_0, 3)]...)
# println("dimension of 0s (obs): ", size(y_0s))

# y_st_0 = zscore(y_0s, 1)
# mle = MultivariateStats.fit(PPCA, y_st_0; maxoutdim=2)
# M_mle = projection(mle)
# M_mle_k2 = M_mle[:, 1:2]
# x_0s_k2 = M_mle_k2' * y_st_0
# scatter(x_0s_k2[2, :], x_0s_k2[1, :], c=:blue, label="Digit 0 (MLE-std, K2)", ms=0.8, msw=0,
# legend = :topright, xlabel="PC 1", ylabel="PC 2")
# scatter!(x_1s_k2[2, :], x_1s_k2[1, :], c=:red, label="Digit 1 (MLE-std, K2)", ms=0.8, msw=0,
# legend = :topright, xlabel="PC 1", ylabel="PC 2")


# C, σ² = vb_ppca_k2(y_st_0, 500, false, debug=true)
# M_vb = svd(C).U
# x_0s_vb2 = M_vb' * y_st_0
# scatter(x_0s_vb2[1, :], x_0s_vb2[2, :], c=:violet, label="Digit 0 (VB-std)", ms=0.8, msw=0, 
# legend = :topright, xlabel="PC 1", ylabel="PC 2")
# scatter!(x_1s_vb2[1, :], x_1s_vb2[2, :], c=:black, label="Digit 1 (VB-std)", ms=0.8, msw=0, 
# legend = :topright, xlabel="PC 1", ylabel="PC 2")
