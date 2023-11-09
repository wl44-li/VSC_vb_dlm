include("kl_optim.jl")
begin
    using CSV
    using DataFrames
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

#main()