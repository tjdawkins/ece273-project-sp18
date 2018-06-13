function [ B, B0 ] = svm_dual( X, y, c, kernel)
%SVM_DUAL Summary of this function goes here
%   Detailed explanation goes here
    if ~exist('kernel')
        kernel = 0;
    end
    
    n = size(X,1);
    d = size(X,2);
    
    % Kernel
    switch kernel
        case 'rbf'
            XX1 = repmat(X,d,1);
            XX2 = repmat(reshape(X,d*d,1),1,d);
            XXP = XX1 - XX2;
            XXP = reshape(XXP,n,d*d);
            P = ve
        otherwise
            P = X'*X;
    end
    
    if c == 0
        % Hard Margin
        cvx_begin
            variables a(d) 
            % Constraints
            a >= 0;
            a'*y == 0;
            ay = a.*y;
            minimize(.5*ay'*P*ay - sum(a))
        cvx_end
        
        B = sum(ay'.*X,2);
        [~,i] = max(a);
        B0 = 1/y(i(1)) - B'*X(:,i(1));
        

    else
        % Soft Margin
        cvx_begin
            variables a(d) 
            ay = a.*y;
            % Constraints
            a >= 0;
            a <= c;
            a'*y == 0;
            
            minimize(.5*ay'*P*ay - sum(a))
        cvx_end
        
        B = sum(ay'.*X,2);
        amask = a<=0 & a>=c;
        at = a;
        at(amask) = -inf;
        [~,i] = max(at);
        B0 = 1/y(i(1)) - B'*X(:,i(1));
        z = 1 - y.*(X'*B + B0);
        z(a == 0) = 0;
                

    end
    
    

end

