function [res, time] = naive(X2, S2, y2)
    tic;
    z2 = inv(transpose(X2)*inv(S2)*X2)*transpose(X2)*inv(S2)*y2;    
    time = toc;
    res = {z2};
end