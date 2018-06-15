function [y] = decision_dual(as,ys,b,SV,X,kernel,kparam)
%DECISION_DUAL Summary of this function goes here
%   Detailed explanation goes here
    if ~exist('kernel')
        kernel = 0;
    end
    
    if ~exist('kparam')
        kparam = 0;
    end
    
    n = size(X,1);
    d = size(SV,2);
    
    % Kernel
    switch kernel
        case 'rbf'            
            p = kernel_rbf(SV,X,kparam);
        otherwise
            p = kernel_induced(X,SV);
    end
    
    ay = as .* ys;
    y = sign(p * ay + b); 

end

