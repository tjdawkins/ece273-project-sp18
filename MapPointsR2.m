function [ x_r2 ] = MapPointsR2( x_rn, B, B0 )
%MAPPOINTSR2 Summary of this function goes here
%   Detailed explanation goes here

m = B'*x_rn + B0;
d = length(x_rn);
t1 = x_rn(1:d/2);
t2 = x_rn(d/2+1:end);

x_r2 = [sum(t1) ; sum(t2)];
x_r2 = m*x_r2;


end

