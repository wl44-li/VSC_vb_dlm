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

train_y, train_labels = MNIST(split=:train)[:]
labels = unique(train_labels)

""" Optional:
-- take 1/10 of the trainning data to test quicker
"""
# train_y, train_labels = train_y[:, :, 1:6000], train_labels[1:6000]

T = size(train_y, 3)
y = hcat([vcat(Float64.(train_y[:, :, t])...) for t in 1:T]...)

# Standardise
y = zscore(y, 1)

# pca = MultivariateStats.fit(PCA, y; maxoutdim=2)
# M = projection(pca)

# M_ = MultivariateStats.fit(PPCA, y; maxoutdim=2)
# M = projection(M_)

"""DEBUG zero-forcing VB-PPCA
-- package :em, :bayes option will not run
-- package default mle option agrees with standard pca

Consider reduce EM iterations (early stop),

Consider standardise data? (get rid of the 0s from white pixels)
"""
C = vb_ppca_k2(y, true)
M = svd(C).U

function compareDigits(dA,dB)
    imA = train_y[:, :, findall(x -> x == dA, train_labels)]
    imB = train_y[:, :, findall(x -> x == dB, train_labels)]

    yA = hcat([vcat(float.(imA[:, :, t])...) for t in 1:size(imA, 3)]...)
    yB = hcat([vcat(float.(imB[:, :, t])...) for t in 1:size(imB, 3)]...)

    xA, xB = M'*yA, M'*yB
    default(ms=0.8, msw=0, xlims=(-5,12.5), ylims=(-7.5,7.5),
            legend = :topright, xlabel="PC 1", ylabel="PC 2")
    scatter(xA[1,:],xA[2,:], c=:red,  label="Digit $(dA)")
    scatter!(xB[1,:],xB[2,:], c=:blue, label="Digit $(dB)")
end

plots = []

for k in 1:5
    push!(plots,compareDigits(2k-2,2k-1))
end

p = plot(plots..., size = (1600, 1000), margin = 5mm)
title!(p, "PPCA")
display(p)