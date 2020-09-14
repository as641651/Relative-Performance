function [out] = operand_generator()
    import MatrixGenerator.*;
    out{ 1 } = generate([1000,500], Shape.General(), Properties.Random([-1, 1]));
    out{ 2 } = generate([1000,1], Shape.General(), Properties.Random([-1, 1]));
end