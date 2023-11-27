include("kl_optim.jl")
begin
    using CSV
    using DataFrames
    using StateSpaceModels
end

function load_csv_to_matrix(file_path::String)
    # Read the CSV file into a DataFrame
    df = CSV.File(file_path) |> DataFrame
    y = df[!, end]
    y = reshape(y, (1, length(y)))  # Reshape the vector into a 1×n matrix
    return y
end

function get_Nile()
    # Specify the path to the CSV file (backup)
    # Other dataset available from StateSpaceModels: https://lampspuc.github.io/StateSpaceModels.jl/latest/manual/#Datasets
    file_path = "./_data/Nile.csv"
    y = load_csv_to_matrix(file_path)
    return y
end

# Not a suitable example 
function get_log_FinTraff()
    df = CSV.File(StateSpaceModels.VEHICLE_FATALITIES) |> DataFrame
    log_ff = log.(df.ff)
    return df, log_ff
end


""" To-Do: fix gibbs, vb for Finland traffic data
function test_traffic()
	df_traff, log_y = get_log_FinTraff()
	println(size(log_y))
	p = plot(df_traff.date, log_y, label="log of Finland traffic fatalities")
	display(p)

	model = LocalLinearTrend(log_y)
	StateSpaceModels.fit!(model)
	print_results(model)

	p_s = plot(df_traff.date, get_smoothed_state(model)[:, 2], label="slope (MLE)")
	display(p_s)

	# test_gibbs(reshape(log_y, 1, :))
	# A_lg = [1.0 1.0; 0.0 1.0]
	# C_lg = [1.0 0.0]
	# K = size(A_lg, 1)
	# prior = HPP_D(2, 1e-4, 2, 1e-4, zeros(K), Matrix{Float64}(I * 1e7, K, K))
	
	# @time R, Q, elbos, Q_gam = vbem_lg_c(reshape(log_y, 1, :), A_lg, C_lg, prior, false, 20, init="fixed", debug=false)
	# μs_f, σs_f2, A_s, Rs, _ = forward_(reshape(log_y, 1, :), A_lg, C_lg, R, Q, prior)
	# μs_s, _, _ = backward_(A_lg, μs_f, σs_f2, A_s, Rs)
	
	# println("\nVB q(R):")
	# show(stdout, "text/plain", R)
	# println("\n\nVB q(Q):")
	# show(stdout, "text/plain", Q)
	# p_s = plot(df_traff.date, μs_s[:, 2], label="slope (VB)")
	# display(p_s)
end

"""

#main()