using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm67(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    start = time_ns()

    # cost: 2.04e+08 FLOPs
    # X2: ml0, full, S2: ml1, full, y2: ml2, full
    # (L21 L21^T) = S2
    LinearAlgebra.LAPACK.potrf!('L', ml1)

    # X2: ml0, full, y2: ml2, full, L21: ml1, lower_triangular
    ml3 = Array{Float64}(undef, 250, 500)
    blascopy!(250*500, ml0, 1, ml3, 1)
    # tmp33 = (L21^-1 X2)
    trsm!('L', 'L', 'N', 'N', 1.0, ml1, ml0)

    # X2: ml3, full, y2: ml2, full, L21: ml1, lower_triangular, tmp33: ml0, full
    ml4 = Array{Float64}(undef, 250, 500)
    blascopy!(250*500, ml0, 1, ml4, 1)
    # tmp34 = (L21^-T tmp33)
    trsm!('L', 'L', 'T', 'N', 1.0, ml1, ml0)

    # X2: ml3, full, y2: ml2, full, L21: ml1, lower_triangular, tmp33: ml4, full, tmp34: ml0, full
    ml5 = Array{Float64}(undef, 500)
    # tmp57 = (tmp34^T y2)
    gemv!('T', 1.0, ml0, ml2, 0.0, ml5)

    # X2: ml3, full, L21: ml1, lower_triangular, tmp33: ml4, full, tmp57: ml5, full
    # tmp33 = (L21^-1 X2)
    trsm!('L', 'L', 'N', 'N', 1.0, ml1, ml3)

    # tmp33: ml3, full, tmp57: ml5, full
    ml6 = Array{Float64}(undef, 500, 250)
    # tmp48 = tmp33^T
    transpose!(ml6, ml3)

    # tmp57: ml5, full, tmp48: ml6, full
    ml7 = Array{Float64}(undef, 500, 500)
    # tmp27 = (tmp48 tmp48^T)
    syrk!('L', 'N', 1.0, ml6, 0.0, ml7)

    # tmp57: ml5, full, tmp27: ml7, symmetric_lower_triangular
    # (L35 L35^T) = tmp27
    LinearAlgebra.LAPACK.potrf!('L', ml7)

    # tmp57: ml5, full, L35: ml7, lower_triangular
    # tmp59 = (L35^-1 tmp57)
    trsv!('L', 'N', 'N', ml7, ml5)

    # L35: ml7, lower_triangular, tmp59: ml5, full
    # tmp26 = (L35^-T tmp59)
    trsv!('L', 'T', 'N', ml7, ml5)

    # tmp26: ml5, full
    # z2 = tmp26

    finish = time_ns()
    return (tuple(ml5), (finish-start)*1e-9)
end