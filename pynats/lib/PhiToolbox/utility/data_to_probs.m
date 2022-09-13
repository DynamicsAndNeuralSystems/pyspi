function probs = data_to_probs(X, params, options)
% Compute probability distributions for time series data X
%
%-----------------------------------------------------------------------
% INPUTS:
%           X: time series data in the form (units X time)
%
%           params: parameters used for estimating probability distributions
%           (covariances in the case of Gaussian distribution) from time
%           series data
%
%           params.tau: time lag between past and present states
%           params.number_of_states: number of states (only for the discrete case)
%           
%           options: options for computing phi and for MIP search
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
% OUTPUTS:
%           probs: probability distributions used for computing phi
%
%           In the Gaussian case
%               When options.type_of_phi is 'MI1'
%                  probs.Cov_X: covariance of data X
%               When options.type_of_phi is NOT 'MI1'
%                  probs.Cov_X: covariance of data X (past, t-tau)
%                  probs.Cov_XY: cross-covariance of X (past, t-tau) and Y (present, t)
%                  probs.Cov_Y: covariance of data Y (present, t)
%           In the discrete case
%               When options.type_of_phi is 'MI1'
%                  probs.past: probability distribution of past state (X^t-tau)
%                  probs.joint: joint distribution of X^t (present) and X^(t-\tau) (past)
%                  probs.present: probability distribution of present state (X^t-tau)
%               When options.type_of_phi is NOT 'MI1'
%                  probs.p: probability distribution of X
%
%
%-----------------------------------------------------------------------


tau = params.tau;

switch options.type_of_phi
    case 'MI1'
        isjoint = 0;
    otherwise
        isjoint = 1;
end

switch options.type_of_dist
    case 'Gauss'
        probs = Cov_comp(X, tau, isjoint);
    case 'discrete'
        number_of_states = params.number_of_states;
        probs = est_p(X, number_of_states, tau, isjoint);
        probs.number_of_states = number_of_states;
    otherwise
        error('type_of_dist must be ''Guass'' or ''discrete''.')
end

probs.number_of_elements = size(X,1);

end