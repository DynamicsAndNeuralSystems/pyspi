function [phi_G, Cov_E_p, A_p] = phi_G_Gauss_LBFGS( Cov_X, Cov_E, A, Z, normalization )
% Calculate integrated information "phi_G" based on information geometry 
% with a combination of an interative method and a quasi-Newton (LBFGS) method. 
% 
% See Oizumi et al., 2016, PNAS for the details of phi_G
% http://www.pnas.org/content/113/51/14817.full
% 
% The code assumes a vector AutoRegressive (VAR) model Y = AX+E,
% where X and Y are the past and present states, A is the connectivity matrix, and E is Gaussian random variables. 
% 
% 
% INPUTS:
%     Cov_X: equal time covariace of X. n by n matrix (n is the number of variables).
%     Cov_E: covariance of noise E. n by n matrix. 
%     A: connectivity (autoregressive) matrix (n by n).
%     Z: partition
%         - 1 by n matrix. Each element value indicates the group number to which the element belongs.
%         - Ex.1:  (1:n) (atomic partition)
%         - Ex.2:  [1, 2,2,2, 3,3, ..., K,K] (K is the number of groups) 
%         - Ex.3:  [3, 1, K, 2, 2, ..., K, 2] (Groups don't have to be sorted in ascending order)
%     normalization: 
%         0: without normalization by Entropy (default)
%         1: with normalization by Entropy
%     
% OUTPUTS:
%     phi_G: integrated information based on information geometry
%     Cov_E_p: covariance of noise E in the disconnected model
%     A_p: connectivity matrix in the disconnected model
%
%
% Masafumi Oizumi, 2016
% Jun Kitazono, 2017
%
% Modification History
%   - made the code able to receive any partition as input (J. Kitazono)
%   - changed the optimization method from steepest descent to an interative method and a quasi-Newton (LBFGS) method. 
%   (J. Kitazono)
%
% This code uses an open optimization toolbox "minFunc" written by M. Shmidt.
% M. Schmidt. minFunc: unconstrained differentiable multivariate optimization in Matlab. 
% http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html
% [Copyright 2005-2015 Mark Schmidt. All rights reserved]

addpath(genpath('minFunc_2012'))

if nargin < 4
    Z = 1: 1: N;
end
if nargin < 5
    normalization = 0;
end

n = size(Cov_X,1);

n_c = max(Z); % number of groups
M_cell = cell(n_c,1);
for i=1: n_c
    M_cell{i} = find(Z==i);
end

% set initial values of the connectivity matrix in the disconnected model
A_p = zeros(n,n);
nnz_A_p = 0;
for i=1: n_c
     M = M_cell{i};
     A_p(M,M) = A(M,M);
     nnz_A_p = nnz_A_p + length(M)^2;
end

iter_max = 10000;
error = 10^-10;

% set options of minFunc
Options.Method = 'lbfgs';
Options.progTol = 10^-10;
Options.MaxFunEvals = 4000;
Options.MaxIter = 2000;
Options.display = 'off';
% Options.useMex = 0;

Cov_E_p = Cov_E;
for iter=1: iter_max
    Cov_E_p_past = Cov_E_p;
    
    x = A_p2vec(A_p, nnz_A_p, M_cell);

    f = @(x)phi_G_grad_Ap( x, Cov_E_p, Cov_X, Cov_E, A, M_cell );
    [x, ~, ~, ~] = minFunc(f, x, Options);
    
    A_p = vec2A_p( x, n, M_cell );
    
    Cov_E_p = Cov_E + (A-A_p)*Cov_X*(A-A_p)';
    
    phi_update = logdet(Cov_E_p) - logdet(Cov_E_p_past);
    if abs(phi_update) < error
        break;
    end
    
end

phi_G = 1/2*(logdet(Cov_E_p)-logdet(Cov_E));
% disp(['iter: ', num2str(iter), ', phi: ', num2str(phi_G)])

if normalization == 1
    H_p = zeros(N_c,1);
    for i=1: N_c
        M = M_cell{i};
        Cov_X_p = Cov_X(M,M); 
        H_p(i) = H_gauss(Cov_X_p);
    end
    if N_c == 1
        phi_G = phi_G/H_p(1);
    else
        phi_G = phi_G/( (N_c-1)*min(H_p) );
    end
end

end

function [phi_IG, D_A_vec] = phi_G_grad_Ap( x, Cov_E_p, Cov_X, Cov_E, A, partition_cell )


N = size(Cov_X,1);

A_p = zeros(size(A));
idx_st = 0;
for i = 1:length(partition_cell)
    M = partition_cell{i};
    nnz_cell_i = length(M);
    idx_end = nnz_cell_i^2;
    A_p(M,M) = reshape(x(idx_st + (1:idx_end)), [nnz_cell_i, nnz_cell_i]);
    idx_st = idx_st + idx_end;
end

R = [Cov_X Cov_X*A'; A*Cov_X Cov_E+A*Cov_X*A'];
Rd = [Cov_X Cov_X*A_p'; A_p*Cov_X Cov_E_p+A_p*Cov_X*A_p'];
Rd_inv = [inv(Cov_X)+A_p'/Cov_E_p*A_p -A_p'/Cov_E_p; -Cov_E_p\A_p inv(Cov_E_p)];

TR = trace(R*Rd_inv);
phi_IG = 1/2*(-logdet(R) + TR + logdet(Rd) - 2*N);


A_diff = A_p - A;

D_A = 2*Cov_E_p\A_diff*Cov_X;
D_A_vec = zeros(length(x),1);
idx_st = 0;
for i = 1:length(partition_cell)
    M = partition_cell{i};
    nnz_cell_i = length(M);
    idx_end = nnz_cell_i^2;
    D_A_vec(idx_st + (1:idx_end)) = reshape(D_A(M,M), [idx_end, 1]);
    idx_st = idx_st + idx_end;
end


end

function A_p = vec2A_p( x, N, partition_cell )
% tramsform vector to matrix

A_p = zeros(N,N);
idx_st = 0;
for i = 1:length(partition_cell)
    M = partition_cell{i};
    nnz_cell_i = length(M);
    idx_end = nnz_cell_i^2;
    
    A_p(M,M) = reshape(x(idx_st+(1:idx_end)), [nnz_cell_i, nnz_cell_i]);
    idx_st = idx_st + idx_end;
end


end

function x = A_p2vec( A_p, nnz_A_p, partition_cell )
% transform matrix to vector

x = zeros(nnz_A_p,1);
idx_st = 0;
for i = 1:length(partition_cell)
    M = partition_cell{i};
    idx_end = length(M)^2;
    x(idx_st + (1:idx_end)) = reshape(A_p(M,M), [idx_end 1]);
    idx_st = idx_st + idx_end;
end


end

