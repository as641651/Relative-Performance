using LinearAlgebra.BLAS
using LinearAlgebra

function naive(X1::Array{Float64,2}, y1::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    start = time_ns()

    b1 = inv(transpose(X1)*X1)*transpose(X1)*y1;

    finish = time_ns()
    return (tuple(b1), (finish-start)*1e-9)
end