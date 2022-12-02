function phi = phi_comp_probs(type_of_dist, type_of_phi, Z, probs, options)
% Compute phi from probability distributions
%
% INPUTS: 
%           type_of_dist:
%              'Gauss': Gaussian distribution
%              'dis': discrete probability distribution
%           type_of_phi:
%              'SI': phi_H, stochastic interaction
%              'Geo': phi_G, information geometry version
%              'star': phi_star, based on mismatched decoding
%              'MI': Multi (Mutual) information, I(X_1, Y_1; X_2, Y_2)
%              'MI1': Multi (Mutual) information. I(X_1; X_2). (IIT1.0)
%           Z: partition
%         - 1 by n matrix. Each element indicates the group number. 
%         - Ex.1:  (1:n) (atomic partition)
%         - Ex.2:  [1, 2,2,2, 3,3, ..., K,K] (K is the number of groups) 
%         - Ex.3:  [3, 1, K, 2, 2, ..., K, 2] (Groups don't have to be sorted in ascending order)
%           probs: probability distributions for computing phi
%           
%           In the Gaussian case
%               probs.Cov_X: covariance of data X (past, t-tau)
%               probs.Cov_XY: cross-covariance of X (past, t-tau) and Y (present, t)
%               probs.Cov_Y: covariance of data Y (present, t)
%           In the discrete case
%               probs.past: probability distribution of past state (X^t-tau)
%               probs.joint: joint distribution of X^t (present) and X^(t-\tau) (past)
%               probs.present: probability distribution of present state (X^t-tau)
%
%               probs.p: probability distribution of X (only used for MI)
%
%           options.normalization (available only for Gaussian dist.)
%              0: without normalization by Entropy (default)
%              1: with normalization by Entropy
%           options.phi_G_OptimMethod (available only for Gaussian dist.)
%              'AL': Augmented Lagrangian
%              'LI': a combination of LBFGS method and Iterative method
%
%
% OUTPUT:
%           phi: integrated information

switch type_of_dist
    case 'Gauss'
        phi = phi_Gauss( type_of_phi, Z, probs, options);
    case 'discrete'
        phi = phi_dis(type_of_phi, Z, probs);
end

end