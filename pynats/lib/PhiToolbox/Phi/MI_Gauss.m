function MI = MI_Gauss(Cov_X,Cov_XY,Cov_Y,Z, normalization)
% Calculate Multi (Mutual) Information given covariance of/between data at past and
% present 
%
% (Ay, 2001; Ay, 2015, Entropy; Barrett & Seth, 2011, PLoS Comp Biol)  
%
% INPUTS:   
%           Cov_X: covariance of data X (past, t-tau)
%           Cov_XY: cross-covariance of X (past, t-tau) and Y (present, t)
%           Cov_Y: covariance of data Y (present, t)
%           Z: partition of each channel (default: atomic partition)
%           normalization: 
%              0: without normalization by Entropy (default)
%              1: with normalization by Entropy
%
% OUTPUT:
%           MI: multi information among subsystems. When the partition is a
%           bi-partition, this becomes mutual information MI(X_1,Y_1; X_2,Y_2).
%
%  Jun Kitazono, 2017

N = size(Cov_X,1); % number of channels
if nargin < 3 || isempty(Cov_Y)
    Cov_Y = Cov_X;
end
if nargin < 4 || isempty(Z)
    Z = 1: 1: N;
end
if nargin < 5 || isempty(normalization)
    normalization = 0;
end

H = H_gauss([Cov_X, Cov_XY; Cov_XY', Cov_Y]);

%%
N_c = max(Z); % number of clusters
M_cell = cell(N_c,1);
for i=1: N_c
    M_cell{i} = find(Z==i);
end

H_p_joint = zeros(N_c,1);

for i=1: N_c
    M = M_cell{i};
    Cov_X_p = Cov_X(M,M);
    Cov_Y_p = Cov_Y(M,M);
    Cov_XY_p = Cov_XY(M,M);
    
    Cov_p = [Cov_X_p, Cov_XY_p; Cov_XY_p', Cov_Y_p];
    H_p_joint(i) = H_gauss(Cov_p);
end

MI = sum(H_p_joint) - H;

if normalization == 1
    H_p = zeros(N_c,1);
    for i=1: N_c
        M = M_cell{i};
        Cov_X_p = Cov_X(M,M); 
        H_p(i) = H_gauss(Cov_X_p);
    end
    if N_c == 1
        MI = MI/H_p(1);
    else
        MI = MI/( (N_c-1)*min(H_p) );
    end
end


end