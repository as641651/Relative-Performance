using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm94(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    start = time_ns()

    # cost: 2.04e+08 FLOPs
    # X2: ml0, full, S2: ml1, full, y2: ml2, full
    # (L21 L21^T) = S2
    LinearAlgebra.LAPACK.potrf!('L', ml1)

    # X2: ml0, full, y2: ml2, full, L21: ml1, lower_triangular
    # tmp51 = (L21^-1 y2)
    trsv!('L', 'N', 'N', ml1, ml2)

    # X2: ml0, full, L21: ml1, lower_triangular, tmp51: ml2, full
    ml3 = Array{Float64}(undef, 250, 500)
    blascopy!(250*500, ml0, 1, ml3, 1)
    # tmp33 = (L21^-1 X2)
    trsm!('L', 'L', 'N', 'N', 1.0, ml1, ml0)

    # X2: ml3, full, L21: ml1, lower_triangular, tmp51: ml2, full, tmp33: ml0, full
    ml4 = Array{Float64}(undef, 500)
    # tmp57 = (tmp33^T tmp51)
    gemv!('T', 1.0, ml0, ml2, 0.0, ml4)

    # X2: ml3, full, L21: ml1, lower_triangular, tmp33: ml0, full, tmp57: ml4, full
    ml5 = Array{Float64}(undef, 250, 500)
    blascopy!(250*500, ml3, 1, ml5, 1)
    # tmp33 = (L21^-1 X2)
    trsm!('L', 'L', 'N', 'N', 1.0, ml1, ml3)

    # X2: ml5, full, L21: ml1, lower_triangular, tmp33: ml3, full, tmp57: ml4, full
    # tmp33 = (L21^-1 X2)
    trsm!('L', 'L', 'N', 'N', 1.0, ml1, ml5)

    # tmp33: ml5, full, tmp57: ml4, full
    ml6 = Array{Float64}(undef, 500, 500)
    # tmp27 = (tmp33^T tmp33)
    syrk!('L', 'T', 1.0, ml5, 0.0, ml6)

    # tmp57: ml4, full, tmp27: ml6, symmetric_lower_triangular
    # (L35 L35^T) = tmp27
    LinearAlgebra.LAPACK.potrf!('L', ml6)

    # tmp57: ml4, full, L35: ml6, lower_triangular
    # tmp59 = (L35^-1 tmp57)
    trsv!('L', 'N', 'N', ml6, ml4)

    # L35: ml6, lower_triangular, tmp59: ml4, full
    # tmp26 = (L35^-T tmp59)
    trsv!('L', 'T', 'N', ml6, ml4)

    # tmp26: ml4, full
    # z2 = tmp26

    finish = time_ns()
    return (tuple(ml4), (finish-start)*1e-9)
end