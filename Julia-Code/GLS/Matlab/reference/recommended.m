function [res, time] = recommended(X2, S2, y2)
    tic;
    z2 = ((transpose(X2)*((S2)\X2))\transpose(X2))*((S2)\y2);    
    time = toc;
    res = {z2};
end