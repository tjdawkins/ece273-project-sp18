function [ B, B0, SV, ys, z ] = svm_primal( X, y, C )
%SVM_PRIMAL Summary of this function goes here
%   Detailed explanation goes here

    n = size(X,1);
    z = 0;

    if C == 0
        % Hard Margin
        cvx_begin
            variables B(n) B0
            y' .* (B'*X + B0) >= 1;    
            minimize(B'*B)
        cvx_end
       
       
    else
        % Soft Margin
        cvx_begin
            variables B(n) B0 z(length(y))
            y' .* (B'*X + B0) + z' >=1;
            z >= 0;
            minimize(B'*B + C*sum(z));
        cvx_end
        
    end
    S_mask = abs(abs(B'*X+B0) -1)<1e-6;
    SV = X(:,S_mask);
    ys = y(S_mask);
end

