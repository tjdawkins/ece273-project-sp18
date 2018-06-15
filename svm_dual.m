function [ B, B0, as, SV, ys, z ] = svm_dual( X, y, c, kernel, kparam)
%SVM_DUAL Summary of this function goes here
%   Detailed explanation goes here
    if ~exist('kernel','var')
        kernel = 0;
    end
    
    if ~exist('kparam','var')
        kparam = 0;
    end
    
    n = size(X,1);
    d = size(X,2);
    delta = 5e-6;
    
    % Kernel
    switch kernel
        case 'rbf'
            P = kernel_rbf(X,X,kparam);
        otherwise
            P = kernel_induced(X,X);
    end
    
    if c == 0
        % Hard Margin
        cvx_begin
            variables a(d) 
            ay = a.*y;
            % Constraints
            a >= 0;
            a'*y == 0;
            
            minimize(.5*ay'*P*ay - sum(a))
        cvx_end
        
        B = sum(ay'.*X,2);
        [~,i] = max(a);
        B0 = 1/y(i(1)) - B'*X(:,i(1));
        %c = inf;
        z = 0;
        
    else
        % Soft Margin
        cvx_begin
            variables a(d) 
            ay = a.*y;
            % Constraints
            a >= 0;
            a <= c;
            a'*y == 0;
            
            minimize(.5*(a.*y)'*P*(a.*y) - sum(a))
        cvx_end

        B = sum(ay'.*X,2);
        amask = a <= 0 + delta | a >= c - delta;
        at = a;
        at(amask) = 0;
        %at(amask) = -inf;
        [~,i] = max(at);
        B0 = y(i(1)) - B'*X(:,i(1));
        z = 1 - y.*(X'*B + B0);
        z(a == 0) = 0;
                

    end
    
    amask = a > delta & a < c + delta;
    as = a(amask);
    SV = X(:,amask);
    ys = y(amask);
end

