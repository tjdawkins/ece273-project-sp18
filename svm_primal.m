function [ B, B0, SV, ys, z ] = svm_primal( X, y, C )
%SVM_PRIMAL Summary of this function goes here
%   Detailed explanation goes here

    n = size(X,1);

    if C == 0
        % Hard Margin
        cvx_begin
            variables B(n) B0
            y' .* (B'*X + B0) >= 1;    
            minimize(B'*B)
        cvx_end
       
        z = zeros(size(y));
       
    else
        % Soft Margin
        cvx_begin
            variables B(n) B0 ze(length(y))
            y' .* (B'*X + B0) + ze' >= 1;
            ze >= 0;
            minimize(B'*B + C*sum(ze));
        cvx_end
        
    end
    S_mask = abs(abs(B'*X+B0) -1)<1e-6;
    SV = X(:,S_mask);
    ys = y(S_mask);
    z = ze;
end

