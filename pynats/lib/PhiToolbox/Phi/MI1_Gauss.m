function MI1 = MI1_Gauss(Cov_X,Z,normalization)
% Calculate Multi (Mutual) Information given covariance of data
%
% (Ay, 2001; Ay, 2015, Entropy; Barrett & Seth, 2011, PLoS Comp Biol)  
%
% INPUTS:   
%           Cov_X: covariance of data X
%           Z: partition of each channel (default: atomic partition)
%           normalization: 
%              0: without normalization by Entropy (default)
%              1: with normalization by Entropy
%
% OUTPUT:
%           MI1: multi information among subsystems. When the partition is a
%           bi-partition, this becomes mutual information MI(X_1; X_2).
%
%  Jun Kitazono, 2017

N = size(Cov_X,1); % number of channels
if nargin < 2 || isempty(Z)
    Z = 1: 1: N;
end
if nargin < 3 || isempty(normalization)
    normalization = 0;
end

H = H_gauss(Cov_X);

%%
N_c = max(Z); % number of clusters
H_p = zeros(N_c,1);

for i=1: N_c
    M = find(Z==i);
    Cov_X_p = Cov_X(M,M);

    H_p(i) = H_gauss(Cov_X_p);
end

MI1 = sum(H_p) - H;
if normalization == 1
    if N_c == 1
        MI1 = MI1/H_p(1);
    else
        MI1 = MI1/( (N_c-1)*min(H_p) );
    end
end


end