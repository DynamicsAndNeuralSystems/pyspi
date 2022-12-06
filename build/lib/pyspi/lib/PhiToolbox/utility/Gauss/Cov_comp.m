function probs = Cov_comp(X,tau,isjoint)
%: Compute covarance matrices from time series data X
%
%-----------------------------------------------------------------------
% INPUTS:
%           X: time series data in the form (units X time)
%           tau: time lag between past and present states
%           isjoint: whether or not joint covraiance matrices are computed
%
% OUTPUT:
%           probs: coraince matrices
%               When isjoint == 1
%                  probs.Cov_X: covariance of data X (past, t-tau)
%                  probs.Cov_XY: cross-covariance of X (past, t-tau) and Y (present, t)
%                  probs.Cov_Y: covariance of data Y (present, t)
%               When isjoint == 0
%                  probs.Cov_X: covariance of data X
%
%-----------------------------------------------------------------------

if nargin < 3
    isjoint = 1;
end

T = size(X,2);

if isjoint
    t_range1 = 1: 1: T-tau;
    t_range2 = 1+tau: 1: T;
    
    X1 = X(:,t_range1);
    X1 = bsxfun(@minus, X1, mean(X1,2)); % subtract mean
    X2 = X(:,t_range2);
    X2 = bsxfun(@minus, X2, mean(X2,2)); % subtract mean
    
    Cov_X = X1*X1'/(T-tau-1); % equal-time covariance matrix at "PAST"
    Cov_Y = X2*X2'/(T-tau-1); % equal-time covariance matrix at "PRESENT"
    Cov_XY = X1*X2'/(T-tau-1); % time-lagged covariance matrix
    
    probs.Cov_Y = Cov_Y;
    probs.Cov_XY = Cov_XY;
else
    X = bsxfun(@minus, X, mean(X,2)); % subtract mean
    Cov_X = X*X'/(T-1); % equal-time covariance matrix
end

probs.Cov_X = Cov_X;

end