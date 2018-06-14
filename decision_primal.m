function [y] = decision_primal(B, B0, X )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

y = sign(B'*X + B0);
y = y';
end

