using LinearAlgebra.BLAS
using LinearAlgebra

"""
    algorithm69(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,1})

Compute
z2 = ((X2^T S2^-1 X2)^-1 X2^T S2^-1 y2).

Requires at least Julia v1.0.

# Arguments
- `ml0::Array{Float64,2}`: Matrix X2 of size 250 x 500 with property FullRank.
- `ml1::Array{Float64,2}`: Matrix S2 of size 250 x 250 with property SPD.
- `ml2::Array{Float64,1}`: Vector y2 of size 250.
"""                    
function algorithm69(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,1})
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
    # tmp33 = (L21^-1 X2)
    trsm!('L', 'L', 'N', 'N', 1.0, ml1, ml3)

    # y2: ml2, full, L21: ml1, lower_triangular, tmp33: ml3, full
    ml4 = Array{Float64}(undef, 500, 250)
    # tmp48 = tmp33^T
    transpose!(ml4, ml3)

    # y2: ml2, full, L21: ml1, lower_triangular, tmp33: ml3, full, tmp48: ml4, full
    # tmp34 = (L21^-T tmp33)
    trsm!('L', 'L', 'T', 'N', 1.0, ml1, ml3)

    # y2: ml2, full, tmp48: ml4, full, tmp34: ml3, full
    ml5 = Array{Float64}(undef, 500)
    # tmp57 = (tmp34^T y2)
    gemv!('T', 1.0, ml3, ml2, 0.0, ml5)

    # tmp48: ml4, full, tmp57: ml5, full
    ml6 = Array{Float64}(undef, 500, 500)
    # tmp27 = (tmp48 tmp48^T)
    syrk!('L', 'N', 1.0, ml4, 0.0, ml6)

    # tmp57: ml5, full, tmp27: ml6, symmetric_lower_triangular
    # (L35 L35^T) = tmp27
    LinearAlgebra.LAPACK.potrf!('L', ml6)

    # tmp57: ml5, full, L35: ml6, lower_triangular
    # tmp59 = (L35^-1 tmp57)
    trsv!('L', 'N', 'N', ml6, ml5)

    # L35: ml6, lower_triangular, tmp59: ml5, full
    # tmp26 = (L35^-T tmp59)
    trsv!('L', 'T', 'N', ml6, ml5)

    # tmp26: ml5, full
    # z2 = tmp26
    return (ml5)
end