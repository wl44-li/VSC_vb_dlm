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
- package :em, :bayes option will not run
- package default mle option agrees with standard pca

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
        C = vb_ppca_k2(y, 10, false)
        M = svd(C).U
    end

    for k in 1:5
        push!(plots, compareDigits(train_y, train_labels, M, 2k-2,2k-1))
    end

    p = plot(plots..., size = (1600, 1000), margin = 5mm)
    title!(p, "$method, $test_prop % data")
    display(p)
end

#test_MNIST(100, true, "vbem")
#test_MNIST()

"""
On-going PPCA-VB Testing
- check ELBO always increasing
- prior choices
- random starts 
- P = 28 x 28 = 784, K = 2
"""

train_y, train_labels = MNIST(split=:train)[:]
train_y, train_labels = train_y[:, :, 1:12000], train_labels[1:12000]

T = size(train_y, 3)
y = hcat([vcat(Float64.(train_y[:, :, t])...) for t in 1:T]...)
y = zscore(y, 1)


C = vb_ppca_k2(y, 50, false, debug=false)
M = svd(C).U

#compare_0(train_y, train_labels, M)
plot_number(train_y, train_labels, 0, M)

plot_number(train_y, train_labels, 1, M)


pca = MultivariateStats.fit(PCA, y; maxoutdim=2)
M_pca = projection(pca)

plot_number(train_y, train_labels, 0, M_pca)
plot_number(train_y, train_labels, 1, M_pca)
