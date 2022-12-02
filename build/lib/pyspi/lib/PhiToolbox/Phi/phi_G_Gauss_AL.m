function [ phi_G, Cov_E_p, A_p ] = phi_G_Gauss_AL( Cov_X, Cov_E, A, Z, normalization )
% Calculate integrated information "phi_G" based on information geometry 
% with an augmented Lagrangian method 
% 
% See Oizumi et al., 2016, PNAS for the details of phi_G
% http://www.pnas.org/content/113/51/14817.full
% 
% The code assumes a vector AutoRegressive (VAR) model Y = AX+E,
% where X and Y are the past and present states, A is the connectivity matrix, and E is Gaussian random variables. 
% 
% INPUTS:
%     Cov_X: equal time covariace of X. n by n matrix (n is the number of variables).
%     Cov_E: covariance of noise E. n by n matrix. 
%     A: connectivity (autoregressive) matrix (n by n).
%     Z: partition
%         - 1 by n matrix. Each element value indicates the group number to which the element belongs.
%         - Ex.1:  (1:n) (atomic partition)
%         - Ex.2:  [1, 2,2,2, 3,3, ..., K,K] (K is the number of groups) 
%         - Ex.3:  [3, 1, K, 2, 2, ..., K, 2] (Groups don't have to be sorted in ascendeing order)
%     normalization: 
%         0: without normalization by Entropy (default)
%         1: with normalization by Entropy
%     
% OUTPUTS:
%     phi_G: integrated information based on information geometry
%     Cov_E_p: covariance of noise E in the disconnected model
%     A_p: connectivity matrix in the disconnected model
%
% Masafumi Oizumi, 2016
% Jun Kitazono, 2017
%
% Modification History
%   - made the code able to receive any partition as input (J. Kitazono)
%   - changed the optimization method from steepest descent to Augmented Lagrangian
%   (J. Kitazono)

maxiter = 100000;
error = 10^-10;
mu = 2;
alpha = 0.9;
gamma = 1.01;

if nargin < 4
    Z = 1: 1: N;
end
if nargin < 5
    normalization = 0;
end

n = size(Cov_X,1);

N_c = max(Z); % number of groups
M_cell = cell(N_c,1);
for i=1: N_c
    M_cell{i} = find(Z==i);
end

% set initial values of the connectivity matrix in disconnected model
A_p = zeros(n,n);
for i=1: N_c
     M = M_cell{i};
     A_p(M,M) = A(M,M);
end
B = A_p;

Lambda = zeros(n,n);

[Q, D_Cov_X] = eig( Cov_X );

Cov_E_p = Cov_E;
for iter = 1:maxiter
    Cov_E_p_past = Cov_E_p;
    A_diff = A-A_p;
    Cov_E_p = Cov_E + A_diff*Cov_X*A_diff';
    
    [P, D_Cov_E_p] = eig(Cov_E_p);
    
    A_p = P* ( ( P'*(B + Lambda/mu )*Q + D_Cov_E_p\P'*A*Q*D_Cov_X/mu ) ./ ...
        (1 + bsxfun(@ldivide, diag(D_Cov_E_p), diag(D_Cov_X)')/mu) ) * Q';

    B = A_p/2 ;
    for i=1: N_c
        M = M_cell{i};
        B(M,M) = B(M,M)*2;
    end
    
    Lambda = Lambda - mu * (A_p-B);
    
    val_constraint = sum((A_p(:)-B(:)).^2);
    if iter > 1
        if val_constraint > alpha*val_constraint_past
            mu = gamma*mu;
        end
    end
    val_constraint_past = val_constraint;   
    
    phi_update = logdet(Cov_E_p) - logdet(Cov_E_p_past);
    if abs(phi_update) < error
        break;
    end

end

phi_G = (logdet(Cov_E_p)-logdet(Cov_E))/2;
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

