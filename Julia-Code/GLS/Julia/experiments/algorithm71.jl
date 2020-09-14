using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm71(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    start = time_ns()

    # cost: 2.04e+08 FLOPs
    # X2: ml0, full, S2: ml1, full, y2: ml2, full
    ml3 = Array{Float64}(undef, 500, 250)
    # tmp47 = X2^T
    transpose!(ml3, ml0)

    # X2: ml0, full, S2: ml1, full, y2: ml2, full, tmp47: ml3, full
    # (L21 L21^T) = S2
    LinearAlgebra.LAPACK.potrf!('L', ml1)

    # X2: ml0, full, y2: ml2, full, tmp47: ml3, full, L21: ml1, lower_triangular
    # tmp48 = (tmp47 L21^-T)
    trsm!('R', 'L', 'T', 'N', 1.0, ml1, ml3)

    # X2: ml0, full, y2: ml2, full, L21: ml1, lower_triangular, tmp48: ml3, full
    # tmp33 = (L21^-1 X2)
    trsm!('L', 'L', 'N', 'N', 1.0, ml1, ml0)

    # y2: ml2, full, L21: ml1, lower_triangular, tmp48: ml3, full, tmp33: ml0, full
    ml4 = Array{Float64}(undef, 500, 250)
    # tmp48 = tmp33^T
    transpose!(ml4, ml0)

    # y2: ml2, full, L21: ml1, lower_triangular, tmp48: ml4, full
    ml5 = Array{Float64}(undef, 500, 250)
    blascopy!(500*250, ml4, 1, ml5, 1)
    # tmp32 = (tmp48 L21^-1)
    trsm!('R', 'L', 'N', 'N', 1.0, ml1, ml4)

    # y2: ml2, full, tmp48: ml5, full, tmp32: ml4, full
    ml6 = Array{Float64}(undef, 500, 500)
    # tmp27 = (tmp48 tmp48^T)
    syrk!('L', 'N', 1.0, ml5, 0.0, ml6)

    # y2: ml2, full, tmp32: ml4, full, tmp27: ml6, symmetric_lower_triangular
    # (L35 L35^T) = tmp27
    LinearAlgebra.LAPACK.potrf!('L', ml6)

    # y2: ml2, full, tmp32: ml4, full, L35: ml6, lower_triangular
    ml7 = Array{Float64}(undef, 500)
    # tmp39 = (tmp32 y2)
    gemv!('N', 1.0, ml4, ml2, 0.0, ml7)

    # L35: ml6, lower_triangular, tmp39: ml7, full
    # tmp41 = (L35^-1 tmp39)
    trsv!('L', 'N', 'N', ml6, ml7)

    # L35: ml6, lower_triangular, tmp41: ml7, full
    # tmp26 = (L35^-T tmp41)
    trsv!('L', 'T', 'N', ml6, ml7)

    # tmp26: ml7, full
    # z2 = tmp26

    finish = time_ns()
    return (tuple(ml7), (finish-start)*1e-9)
end