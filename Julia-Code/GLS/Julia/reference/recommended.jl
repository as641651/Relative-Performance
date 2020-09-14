using LinearAlgebra.BLAS
using LinearAlgebra

function recommended(X2::Array{Float64,2}, S2::Symmetric{Float64,Array{Float64,2}}, y2::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    start = time_ns()

    z2 = ((transpose(X2)*((S2)\X2))\transpose(X2))*((S2)\y2);

    finish = time_ns()
    return (tuple(z2), (finish-start)*1e-9)
end