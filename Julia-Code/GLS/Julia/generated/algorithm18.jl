using LinearAlgebra.BLAS
using LinearAlgebra

"""
    algorithm18(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,1})

Compute
z2 = ((X2^T S2^-1 X2)^-1 X2^T S2^-1 y2).

Requires at least Julia v1.0.

# Arguments
- `ml0::Array{Float64,2}`: Matrix X2 of size 250 x 500 with property FullRank.
- `ml1::Array{Float64,2}`: Matrix S2 of size 250 x 250 with property SPD.
- `ml2::Array{Float64,1}`: Vector y2 of size 250.
"""                    
function algorithm18(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,1})
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
    # tmp32 = (tmp48 L21^-1)
    trsm!('R', 'L', 'N', 'N', 1.0, ml1, ml3)

    # y2: ml2, full, tmp27: ml4, symmetric_lower_triangular, tmp32: ml3, full
    ml5 = Array{Float64}(undef, 500)
    # tmp39 = (tmp32 y2)
    gemv!('N', 1.0, ml3, ml2, 0.0, ml5)

    # tmp27: ml4, symmetric_lower_triangular, tmp39: ml5, full
    # (L35 L35^T) = tmp27
    LinearAlgebra.LAPACK.potrf!('L', ml4)

    # tmp39: ml5, full, L35: ml4, lower_triangular
    # tmp41 = (L35^-1 tmp39)
    trsv!('L', 'N', 'N', ml4, ml5)

    # L35: ml4, lower_triangular, tmp41: ml5, full
    # tmp26 = (L35^-T tmp41)
    trsv!('L', 'T', 'N', ml4, ml5)

    # tmp26: ml5, full
    # z2 = tmp26
    return (ml5)
end