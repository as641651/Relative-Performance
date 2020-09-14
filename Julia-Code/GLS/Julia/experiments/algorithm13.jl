using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm13(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    start = time_ns()

    # cost: 1.73e+08 FLOPs
    # X2: ml0, full, S2: ml1, full, y2: ml2, full
    # (L21 L21^T) = S2
    LinearAlgebra.LAPACK.potrf!('L', ml1)

    # X2: ml0, full, y2: ml2, full, L21: ml1, lower_triangular
    # tmp33 = (L21^-1 X2)
    trsm!('L', 'L', 'N', 'N', 1.0, ml1, ml0)

    # y2: ml2, full, L21: ml1, lower_triangular, tmp33: ml0, full
    ml3 = Array{Float64}(undef, 500, 250)
    # tmp48 = tmp33^T
    transpose!(ml3, ml0)

    # y2: ml2, full, L21: ml1, lower_triangular, tmp48: ml3, full
    ml4 = Array{Float64}(undef, 500, 500)
    # tmp27 = (tmp48 tmp48^T)
    syrk!('L', 'N', 1.0, ml3, 0.0, ml4)

    # y2: ml2, full, L21: ml1, lower_triangular, tmp48: ml3, full, tmp27: ml4, symmetric_lower_triangular
    # (L35 L35^T) = tmp27
    LinearAlgebra.LAPACK.potrf!('L', ml4)

    # y2: ml2, full, L21: ml1, lower_triangular, tmp48: ml3, full, L35: ml4, lower_triangular
    # tmp32 = (tmp48 L21^-1)
    trsm!('R', 'L', 'N', 'N', 1.0, ml1, ml3)

    # y2: ml2, full, L35: ml4, lower_triangular, tmp32: ml3, full
    ml5 = Array{Float64}(undef, 500)
    # tmp39 = (tmp32 y2)
    gemv!('N', 1.0, ml3, ml2, 0.0, ml5)

    # L35: ml4, lower_triangular, tmp39: ml5, full
    # tmp41 = (L35^-1 tmp39)
    trsv!('L', 'N', 'N', ml4, ml5)

    # L35: ml4, lower_triangular, tmp41: ml5, full
    # tmp26 = (L35^-T tmp41)
    trsv!('L', 'T', 'N', ml4, ml5)

    # tmp26: ml5, full
    # z2 = tmp26

    finish = time_ns()
    return (tuple(ml5), (finish-start)*1e-9)
end