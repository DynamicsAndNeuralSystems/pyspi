function [SI, I, H_cond] = SI_Gauss(Cov_X,Cov_XY,Cov_Y,Z,normalization)
% Calculate stochastic interaction given covariance of data
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
%           SI: stochastic interaction SI(Y|X) 
%           I: mutual information I(X;Y)
%           H: entropy, H(Y)
%
%  Masafumi Oizumi, 2017

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

Cov_Y_X = Cov_cond(Cov_Y,Cov_XY',Cov_X); % conditional covariance matrix
H_cond = H_gauss(Cov_Y_X);

if isinf(H_cond) == 1
    fprintf('Alert: Infinite Entropy\n')
end

if isreal(H_cond) == 0
    fprintf('Alert: Complex Entropy\n')
end

H = H_gauss(Cov_Y); % entropy
I = H - H_cond; % mutual information

%% 
N_c = max(Z); % number of clusters
H_cond_p = zeros(N_c,1);

M_cell = cell(N_c,1);
for i=1: N_c
    M_cell{i} = find(Z==i);
end

for i=1: N_c
    M = M_cell{i};
    Cov_X_p = Cov_X(M,M);
    Cov_Y_p = Cov_Y(M,M);
    Cov_XY_p = Cov_XY(M,M);
    
    Cov_Y_X_p = Cov_cond(Cov_Y_p,Cov_XY_p',Cov_X_p);
    H_cond_p(i) = H_gauss(Cov_Y_X_p);
end


%% stochastic interaction
SI = sum(H_cond_p) - H_cond;

if normalization == 1
    H_p = zeros(N_c,1);
    for i=1: N_c
        M = M_cell{i};
        Cov_X_p = Cov_X(M,M); 
        H_p(i) = H_gauss(Cov_X_p);
    end
    if N_c == 1
        SI = SI/H_p(1);
    else
        SI = SI/( (N_c-1)*min(H_p) );
    end
end

end