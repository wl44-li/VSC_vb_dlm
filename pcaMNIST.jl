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

#test_MNIST(100, true)
#test_MNIST(100, true, "vbem")

train_y, train_labels = MNIST(split=:train)[:]
train_y, train_labels = train_y[:, :, 1:60000], train_labels[1:60000]

# plots_pca = []
# for k in [1, 6, 8, 9]
#     push!(plots_pca, plot_number(train_y, train_labels, k, "pca"))
# end
# p_pca = plot(plots_pca..., size = (1600, 1000), margin = 5mm)
# display(p_pca)

plots_ppca = []
for k in [1, 6, 8, 9]
    push!(plots_ppca, plot_number(train_y, train_labels, k, "ppca"))
end
p_ppca = plot(plots_ppca..., size = (1600, 1000), margin = 5mm)
display(p_ppca)

# T = size(train_y, 3)
# y = hcat([vcat(Float64.(train_y[:, :, t])...) for t in 1:T]...)

# # 1s
# img_1 = train_y[:, :, findall(x -> x == 1, train_labels)]
# y_1s = hcat([vcat(float.(img_1[:, :, t])...) for t in 1:size(img_1, 3)]...)
# println("dimension of 1s (MNIST): ", size(y_1s))

# img_0 = train_y[:, :, findall(x -> x == 0, train_labels)]
# y_0s = hcat([vcat(float.(img_0[:, :, t])...) for t in 1:size(img_0, 3)]...)
# println("dimension of 0s (MNIST): ", size(y_0s))

# """
# Simple Data visualisation experiment, separate 0 and 1 from MNIST
# - compare VB with MLE baseline
# """

# y_st = zscore(y_1s, 1)
# y_st_0 = zscore(y_0s, 1)

function show_pca(y_1, y_0)
    pca = MultivariateStats.fit(PCA, y_1; maxoutdim=2)
    M_pca = projection(pca)
    x_1s = M_pca' * y_1
    p_pca = scatter(x_1s[1, :], x_1s[2, :], series_annotations = text.(1, :top, :blue, 5), label="", ms=0, msw=0, 
    legend = :topleft, xlabel="PC 1", ylabel="PC 2")

    pca_0 = MultivariateStats.fit(PCA, y_0; maxoutdim=2)
    M_pca_0 = projection(pca_0)
    x_0s = M_pca_0' * y_0
    scatter!(p_pca, x_0s[1, :], x_0s[2, :], series_annotations = text.(0, :top, :black, 5), label="", ms=0, msw=0)
    display(p_pca)
end

#show_pca(y_st, y_st_0)

function show_mle(y_st, y_st_0)
    mle_std = MultivariateStats.fit(PPCA, y_st; maxoutdim=2)
    M_mle_k2 = svd(mle_std.W[:, 1:2]).U
    x_1s_k2 = M_mle_k2' * y_st
    p_mle_n1 = scatter(x_1s_k2[1, :], x_1s_k2[2, :] .* (-1), label="", series_annotations = text.(1, :top, :blue, 5), ms=0, msw=0,
    legend = :topleft, xlabel="PC 1", ylabel="PC 2")
    #display(p_mle_n1)

    mle = MultivariateStats.fit(PPCA, y_st_0; maxoutdim=2)
    M_prj_0 = svd(mle.W[:, 1:2]).U
    x_0s_k2 = M_prj_0' * y_st_0
    scatter!(p_mle_n1, x_0s_k2[1, :], x_0s_k2[2, :] .* (-1), label="", series_annotations = text.(0, :top, :black, 5), ms=0, msw=0)
    display(p_mle_n1)
end

#show_mle(y_st, y_st_0)

# C, _, els_1 = vb_ppca_k2(y_st, 1, false, mode="mle")
# M_vb = svd(C).U
# x_1s_vb2 = M_vb' * y_st
# p_vb_n1 = scatter(x_1s_vb2[1, :], x_1s_vb2[2, :], series_annotations = text.(1, :top, :blue, 5), label="", ms=0, msw=0, 
# legend = :topleft, xlabel="PC 1", ylabel="PC 2")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           

# C_0, _, els_0 = vb_ppca_k2(y_st_0, 1, false, mode="mle")
# M_vb_0 = svd(C_0).U
# x_0s_vb2 = M_vb_0' * y_st_0
# scatter!(p_vb_n1, x_0s_vb2[1, :], x_0s_vb2[2, :],  series_annotations = text.(0, :top, :black, 5), label="", ms=0, msw=0, 
# legend = :topleft, xlabel="PC 1", ylabel="PC 2")
# display(p_vb_n1)

function show_vb_mnist_progress(y_st, y_st_0, i_start, i_end, n)
    for i in range(i_start, i_end, step=n)
        C, _, _ = vb_ppca_k2(y_st, i, true, mode="mle")
        M_vb = svd(C).U
        x_1s_vb2 = M_vb' * y_st
        p_vb_n1 = plot()
        p_vb_n1 = scatter(x_1s_vb2[2, :], x_1s_vb2[1, :], c=:blue, label="Digit 1 (VB, std)", ms=1.5, msw=0, 
        legend = :topleft, xlabel="PC 1", ylabel="PC 2")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           

        C_0, _, _ = vb_ppca_k2(y_st_0, i, true, mode="mle")
        M_vb_0 = svd(C_0).U
        x_0s_vb2 = M_vb_0' * y_st_0
        scatter!(p_vb_n1, x_0s_vb2[2, :], x_0s_vb2[1, :], c=:red, label="Digit 0 (VB, std)", ms=1.5, msw=0, 
        legend = :topleft, xlabel="PC 1", ylabel="PC 2")
        display(p_vb_n1)
    end
end

function mnist_vb(y_st, y_st_0, i_start, i_end, n)
    for i in range(i_start, i_end, step=n)

        _, _, _, xs_1, _ = vb_ppca_k2(y_st, i, true, mode="mle")

        _, _, _, xs_0, _ = vb_ppca_k2(y_st_0, i, true, mode="mle")

        p_vb_n1 = plot()
        p_vb_n1 = scatter(xs_1[1, :], xs_1[2, :], c=:blue, label="Digit 1", ms=1.5, msw=0, 
        legend = :topleft, xlabel="PC 1", ylabel="PC 2")
    
        scatter!(p_vb_n1, xs_0[1, :].* (-1), xs_0[2, :] .* (-1), c=:red, label="Digit 0", ms=1.5, msw=0)
        display(p_vb_n1)
    end
end