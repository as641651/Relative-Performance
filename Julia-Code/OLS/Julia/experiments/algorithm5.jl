using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm5(ml0::Array{Float64,2}, ml1::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    start = time_ns()

    # cost: 5.43e+08 FLOPs
    # X1: ml0, full, y1: ml1, full
    ml2 = Array{Float64}(undef, 500, 500)
    # tmp7 = (X1^T X1)
    gemm!('T', 'N', 1.0, ml0, ml0, 0.0, ml2)

    # X1: ml0, full, y1: ml1, full, tmp7: ml2, full
    ml3 = Array{Float64}(undef, 500)
    # tmp15 = (X1^T y1)
    gemv!('T', 1.0, ml0, ml1, 0.0, ml3)

    # tmp7: ml2, full, tmp15: ml3, full
    # (L8 L8^T) = tmp7
    LinearAlgebra.LAPACK.potrf!('L', ml2)

    # tmp15: ml3, full, L8: ml2, lower_triangular
    # tmp17 = (L8^-1 tmp15)
    trsv!('L', 'N', 'N', ml2, ml3)

    # L8: ml2, lower_triangular, tmp17: ml3, full
    # tmp13 = (L8^-T tmp17)
    trsv!('L', 'T', 'N', ml2, ml3)

    # tmp13: ml3, full
    # b1 = tmp13

    finish = time_ns()
    return (tuple(ml3), (finish-start)*1e-9)
end