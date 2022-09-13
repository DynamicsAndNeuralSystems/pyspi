function logdet_X = logdet(X)
%% compute log of determinant of X

n = size(X,1);
Const = exp( sum( log(diag(X)) )/n );

X = X/Const;

logdet_X = log(det(X)) + log(Const)*n;
