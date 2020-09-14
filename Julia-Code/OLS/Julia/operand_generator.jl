using MatrixGenerator

function operand_generator()
    X1::Array{Float64,2} = generate((1000,500), [Shape.General, Properties.Random(-1, 1)])
    y1::Array{Float64,1} = generate((1000,1), [Shape.General, Properties.Random(-1, 1)])
    return (X1, y1,)
end