using Plots
using Distributions

# Function to perform Gibbs Sampling on a bivariate normal distribution
function gibbs_sampling(n_samples)
    samples = zeros(n_samples, 2)
    x, y = 4.0, 4.0
    for i in 1:n_samples
        x = randn() + y  # Sampling from conditional distribution P(X|Y)
        y = randn() + x  # Sampling from conditional distribution P(Y|X)
        samples[i, :] = [x, y]
    end
    return samples
end

function gibbs_plot()
    # Generate samples using Gibbs Sampling
    n_samples = 30
    samples = gibbs_sampling(n_samples)

    # Create a plot to show the zig-zag paths
    p = plot()

    # Overlay the true bivariate normal distribution as contour lines
    x = range(-5, stop=5, length=100)
    y = range(-5, stop=5, length=100)
    z = [pdf(MvNormal([0.0,0.0], [2.0 0.0; 0.0 2.0]), [x[i], y[j]]) for i in 1:length(x), j in 1:length(y)]
    contour!(p, x, y, z, color=:grey, alpha=0.5)

    # Plot the Gibbs samples as blue dots and connect them with red lines
    plot!(p, samples[:,1], samples[:,2], seriestype=:scatter, color=:blue, label="Gibbs Samples")
    plot!(p, samples[:,1], samples[:,2], linecolor=:red, linewidth=0.5, linealpha=0.5, label="Markov Chain")

    # Display the plot
    display(p)
end

gibbs_plot()

function i_projection_plot()
    # Assume a target GMM with 3 components
    weights = [0.2, 0.3, 0.5]
    means = [-5.0, 0.0, 5.0]
    stds = [1.0, 1.0, 1.0]

    # Generate data from the target GMM
    n_samples = 100
    data = []
    for i in 1:n_samples
        z = rand(Categorical(weights))
        x = rand(Normal(means[z], stds[z]))
        push!(data, x)
    end

    # Assume the I-projection leads to a single Gaussian with mean 2.0 and std 1.0
    iproj_mean = 4.8
    iproj_std = 1.0

    # Plot the target GMM and I-projection
    x = range(-10, stop=10, length=200)
    y_true = sum(w * pdf.(Normal(μ, σ), x) for (w, μ, σ) in zip(weights, means, stds))
    y_iproj = pdf.(Normal(iproj_mean, iproj_std), x)

    # Scale down y_true to match the maximum value of y_iproj
    scale_factor = maximum(y_iproj) / maximum(y_true)
    y_true_scaled = y_true * scale_factor

    p = plot(x, y_true_scaled, label="Target PDF", linewidth=2)
    plot!(x, y_iproj, label="I-Projection", linewidth=2, linestyle=:dash, color=:red)
    display(p)
end

i_projection_plot()