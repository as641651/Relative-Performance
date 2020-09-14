using MatrixGenerator

function operand_generator()
    X2::Array{Float64,2} = generate((250,500), [Shape.General, Properties.Random(-1, 1)])
    S2::Symmetric{Float64,Array{Float64,2}} = generate((250,250), [Shape.Symmetric, Properties.SPD])
    y2::Array{Float64,1} = generate((250,1), [Shape.General, Properties.Random(-1, 1)])
    return (X2, S2, y2,)
end