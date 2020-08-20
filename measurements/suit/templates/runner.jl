using Test
using Logging
using MatrixGenerator

using LinearAlgebra.BLAS
BLAS.set_num_threads({num_threads})

{include_operand_generator}
{include_experiments}
#include("operand_generator.jl")
#include("experiments/algorithm0.jl")

function main()
    matrices = operand_generator()

    n = 2000
    rand(n, n)*rand(n, n) # this seems to help to reduce some startup noise

    @info("Running Benchmarks...")
    plotter = Benchmarker.Plot("{result_file}", ["algorithm"; "threads"]);
    {experiments}
    # Benchmarker.add_data(plotter, ["algorithm0"; 24], Benchmarker.measure(2, algo_num, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.finish(plotter);
    @info("Benchmarks complete")
end

main()
