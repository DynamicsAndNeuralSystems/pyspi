function Cov_X_Y = Cov_cond(Cov_X,Cov_XY,Cov_Y)
%% compute the partial covariance of X given Y
% Input
% Cov_X, Cov_Y: covariance of data X and Y
% Cov_XY: cross covariance of X and Y

% Output
% Cov_X_Y: partial covariance of X given Y

if nargin < 3
    Cov_Y = Cov_X;
end

Cov_X_Y = Cov_X - Cov_XY/Cov_Y*Cov_XY';

end
