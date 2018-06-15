function [P] = kernel_rbf(X,Y,g)
%KERNEL_RBF Summary of this function goes here
%   Detailed explanation goes here

    n = size(X,1);
    m = size(Y,2);
    d = size(X,2);
            
    XX1 = repmat(X,m,1);
    XX2 = repmat(reshape(Y,n*m,1),1,d);
    XXP = XX1 - XX2;
    XXP = reshape(XXP,n,d*m);
    P = vecnorm(XXP);
    P = reshape(P,m,d);
    P = exp(-g*P);

end

