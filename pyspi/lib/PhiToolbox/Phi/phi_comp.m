function phi = phi_comp(X, Z, params, options)
% Compute phi from time series data
%
% INPUTS:
%           X: time series data in the form (units X time)
%           Z: partition
%         - 1 by n matrix. Each element indicates the group number. 
%         - Ex.1:  (1:n) (atomic partition)
%         - Ex.2:  [1, 2,2,2, 3,3, ..., K,K] (K is the number of groups) 
%         - Ex.3:  [3, 1, K, 2, 2, ..., K, 2] (Groups don't have to be sorted in ascending order)
%           params: parameters used for estimating probability distributions
%           (covariances in the case of Gaussian distribution) from time
%           series data
%
%           params.tau: time lag between past and present states
%           params.number_of_states: number of states (only for the discrete case)
%           
%           options: options for computing phi
%           
%           options.type_of_dist:
%              'Gauss': Gaussian distribution
%              'discrete': discrete probability distribution
%           options.type_of_phi:
%              'SI': phi_H, stochastic interaction
%              'Geo': phi_G, information geometry version
%              'star': phi_star, based on mismatched decoding
%              'MI': Multi (Mutual) information, I(X_1, Y_1; X_2, Y_2)
%              'MI1': Multi (Mutual) information. I(X_1; X_2). (IIT1.0)
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

probs = data_to_probs(X, params, options);
phi = phi_comp_probs(options.type_of_dist, options.type_of_phi, Z, probs, options);

end