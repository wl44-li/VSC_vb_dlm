include("kl_optim.jl")
begin
    using CSV
    using DataFrames
end

function load_csv_to_matrix(file_path::String)
    # Read the CSV file into a DataFrame
    df = CSV.File(file_path) |> DataFrame

    # Assume you want to extract all columns into a matrix
    # You can adjust the column selection as needed
    y = df[!, end]
    y = reshape(y, (1, length(y)))  # Reshape the vector into a 1×n matrix
    return y
end

function main()
    # Specify the path to the CSV file
    file_path = "./_data/UKgas.csv"

    #file_path = "./_data/Nile.csv"
    # Load the CSV data into a matrix
    y = load_csv_to_matrix(file_path)
    println(size(y))
    println(y)
end

main()