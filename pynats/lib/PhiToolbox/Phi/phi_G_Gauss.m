function [ phi_G, Cov_E_p, A_p ] = phi_G_Gauss( Cov_X, Cov_XY, Cov_Y, Z, method, normalization )
% Calculate integrated information "phi_G" based on information geometry. 
% See Oizumi et al., 2016, PNAS for the details.
% http://www.pnas.org/content/113/51/14817.full
% 
% INPUTS:
%     Cov_X: covariance of data X (past, t-tau). n by n matrix (n is the number of variables).
%     Cov_XY: cross-covariance of X (past, t-tau) and Y (present, t).
%     Cov_Y: covariance of data Y (present, t)
%     Z: partition
%         - 1 by n matrix. Each element value indicates the group number which the element belongs to.
%         - Ex.1:  (1:n) (atomic partition) [default]
%         - Ex.2:  [1, 2,2,2, 3,3, ..., K,K] (K is the number of groups) 
%         - Ex.3:  [3, 1, K, 2, 2, ..., K, 2] (Groups don't have to be sorted in ascending order)
%     method: optimization method, 'AL'|'LI'
%         - 'AL': Augmented Lagrangian
%         - 'LI': a combination of LBFGS method and Iterative method
%     Note: AL tends to be faster than LI when the number of gropus K is small 
%     but is slower when K is large. Thus, by default, this function
%     chooses AL when K<=3 and LI when K>3.
% 
%     normalization: 
%         0: without normalization by Entropy (default)
%         1: with normalization by Entropy
% 
% 
% OUTPUTS:
%     phi_G: integrated information based on information geometry
%     Cov_E_p: covariance of noise E in the disconnected model
%     A_p: connectivity matrix in the disconnected model
%
% Jun Kitazono & Masafumi Oizumi, 2017


% Transform covariance matrices to parameters in AR model, Y = AX+E.
A = Cov_XY'/Cov_X;
Cov_E = Cov_Y - Cov_XY'/Cov_X*Cov_XY;

if nargin < 4 || isempty(Z)
    n = size(Cov_X,1);
    Z = 1:n;
end
if nargin < 5 || isempty(method)
    K = length(unique(Z));
    if K <= 3
        method = 'AL';
    else
        method = 'LI';
    end
end
if nargin < 6 || isempty(normalization)
    normalization = 0;
end

switch method
    case 'AL'
        [phi_G, Cov_E_p, A_p] = phi_G_Gauss_AL( Cov_X, Cov_E, A, Z, normalization );
    case 'LI'
        [phi_G, Cov_E_p, A_p] = phi_G_Gauss_LBFGS( Cov_X, Cov_E, A, Z, normalization );
end


end

