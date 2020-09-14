function [out] = operand_generator()
    import MatrixGenerator.*;
    out{ 1 } = generate([250,500], Shape.General(), Properties.Random([-1, 1]));
    out{ 2 } = generate([250,250], Shape.Symmetric(), Properties.SPD());
    out{ 3 } = generate([250,1], Shape.General(), Properties.Random([-1, 1]));
end