function [x] = gen_data_radial_r2(n,u,s,r)
%FEN_DATA_RADIAL_R2 Summary of this function goes here
%   Detailed explanation goes here

    % Generate Data for RBF 
    px = linspace(u(1)-4*s,u(1)+4*s,1000);
    p = normpdf(px,u(1),s);
    x1 = randpdf(p,px,[n 1]);
    x2 = randpdf(p,px,[n 1]);
    x = [x1'; x2'];
    
    if exist('r','var')
       
        d = vecnorm((x - u));
        x = x(:,d >= sqrt(r));
        
    end
    
    x = sortrows(x')';
    
end

