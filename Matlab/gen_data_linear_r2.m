function [ x ] = gen_data_linear_r2( n, L, s, th, os )
%GEN_DATA_R2 Summary of this function goes here
%   Detailed explanation goes here

    % Generate linearly separable data using log normal pdf
    % https://learnforeverlearn.com/explorelognormal/https://learnforeverlearn.com/explorelognormal/
    R = [cosd(th) -sind(th); sind(th) cosd(th)];    
    px = linspace(-4*s,4*s,1000);
    p = normpdf(px,0,s);
    x2 = randpdf(p,px,[n 1]);
    x1 = linspace(0,L,n)';
    x = [x1'; x2'];
    x = R*(x + os);

end

