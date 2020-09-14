function [res, time] = naive(X1, y1)
    tic;
    b1 = inv(transpose(X1)*X1)*transpose(X1)*y1;    
    time = toc;
    res = {b1};
end