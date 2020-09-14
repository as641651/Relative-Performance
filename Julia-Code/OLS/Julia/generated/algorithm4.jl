using LinearAlgebra.BLAS
using LinearAlgebra

"""
    algorithm4(ml0::Array{Float64,2}, ml1::Array{Float64,1})

Compute
b1 = ((X1^T X1)^-1 X1^T y1).

Requires at least Julia v1.0.

# Arguments
- `ml0::Array{Float64,2}`: Matrix X1 of size 1000 x 500 with property FullRank.
- `ml1::Array{Float64,1}`: Vector y1 of size 1000.
"""                    
function algorithm4(ml0::Array{Float64,2}, ml1::Array{Float64,1})
    # cost: 5.43e+08 FLOPs
    # X1: ml0, full, y1: ml1, full
    ml2 = Array{Float64}(undef, 500)
    # tmp15 = (X1^T y1)
    gemv!('T', 1.0, ml0, ml1, 0.0, ml2)

    # X1: ml0, full, tmp15: ml2, full
    ml3 = Array{Float64}(undef, 500, 500)
    # tmp7 = (X1^T X1)
    gemm!('T', 'N', 1.0, ml0, ml0, 0.0, ml3)

    # tmp15: ml2, full, tmp7: ml3, full
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
    return (ml2)
end