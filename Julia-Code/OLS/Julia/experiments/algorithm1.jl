using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm1(ml0::Array{Float64,2}, ml1::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    start = time_ns()

    # cost: 2.93e+08 FLOPs
    # X1: ml0, full, y1: ml1, full
    ml2 = Array{Float64}(undef, 500)
    # tmp15 = (X1^T y1)
    gemv!('T', 1.0, ml0, ml1, 0.0, ml2)

    # X1: ml0, full, tmp15: ml2, full
    ml3 = Array{Float64}(undef, 500, 500)
    # tmp7 = (X1^T X1)
    syrk!('L', 'T', 1.0, ml0, 0.0, ml3)

    # tmp15: ml2, full, tmp7: ml3, symmetric_lower_triangular
    # (L8 L8^T) = tmp7
    LinearAlgebra.LAPACK.potrf!('L', ml3)

    # tmp15: ml2, full, L8: ml3, lower_triangular
    # tmp17 = (L8^-1 tmp15)
    trsv!('L', 'N', 'N', ml3, ml2)

    # L8: ml3, lower_triangular, tmp17: ml2, full
    # tmp13 = (L8^-T tmp17)
    trsv!('L', 'T', 'N', ml3, ml2)

    # tmp13: ml2, full
    # b1 = tmp13

    finish = time_ns()
    return (tuple(ml2), (finish-start)*1e-9)
end