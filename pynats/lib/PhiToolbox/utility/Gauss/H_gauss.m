function H = H_gauss(Cov_X)
%-----------------------------------------------------------------------
% FUNCTION: H_gauss.m
% PURPOSE: calculate entropy under the gaussian assumption
% 
% INPUTS:
%           Cov_X: covariance of data X
%
% OUTPUT:
%           H: entropy of X
%-----------------------------------------------------------------------

n = size(Cov_X,1);
H = 1/2*logdet(Cov_X) + 1/2*n*log(2*pi*exp(1));

end