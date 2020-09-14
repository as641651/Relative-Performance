using LinearAlgebra.BLAS
using LinearAlgebra

"""
    algorithm0(ml0::Array{Float64,2}, ml1::Array{Float64,1})

Compute
b1 = ((X1^T X1)^-1 X1^T y1).

Requires at least Julia v1.0.

# Arguments
- `ml0::Array{Float64,2}`: Matrix X1 of size 1000 x 500 with property FullRank.
- `ml1::Array{Float64,1}`: Vector y1 of size 1000.
"""                    
function algorithm0(ml0::Array{Float64,2}, ml1::Array{Float64,1})
    # cost: 2.93e+08 FLOPs
    # X1: ml0, full, y1: ml1, full
    ml2 = Array{Float64}(undef, 500, 500)
    # tmp7 = (X1^T X1)
    syrk!('L', 'T', 1.0, ml0, 0.0, ml2)

    # X1: ml0, full, y1: ml1, full, tmp7: ml2, symmetric_lower_triangular
    # (L8 L8^T) = tmp7
    LinearAlgebra.LAPACK.potrf!('L', ml2)

    # X1: ml0, full, y1: ml1, full, L8: ml2, lower_triangular
    ml3 = Array{Float64}(undef, 500)
    # tmp15 = (X1^T y1)
    gemv!('T', 1.0, ml0, ml1, 0.0, ml3)

    # L8: ml2, lower_triangular, tmp15: ml3, full
    # tmp17 = (L8^-1 tmp15)
    trsv!('L', 'N', 'N', ml2, ml3)

    # L8: ml2, lower_triangular, tmp17: ml3, full
    # tmp13 = (L8^-T tmp17)
    trsv!('L', 'T', 'N', ml2, ml3)

    # tmp13: ml3, full
    # b1 = tmp13
    return (ml3)
end