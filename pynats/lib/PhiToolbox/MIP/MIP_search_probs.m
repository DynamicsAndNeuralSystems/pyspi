function [Z_MIP, phi_MIP, Zs, phis] = MIP_search_probs( probs, options )
% Find the Minimum Informamtion Partition from probability distirubtions
%
% INPUTS: 
%           probs: probability distributions for computing phi
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
%                  probs.past: probability distribution of past state X(t-tau)
%                  probs.joint: joint distribution of X(t) (present) and X(t-tau) (past)
%                  probs.present: probability distribution of present state X(t)
%               When options.type_of_phi is NOT 'MI1'
%                  probs.p: probability distribution of X
%
%           options: options for computing phi and for MIP search
%           
%           options.type_of_dist:
%              'Gauss': Gaussian distribution
%              'discrete': discrete probability distribution
%           options.type_of_phi:
%              'SI': phi_H, stochastic interaction
%              'Geo': phi_G, information geometry version (only for 'Gauss')
%              'star': phi_star, based on mismatched decoding
%              'MI': Multi (Mutual) information, I(X_1(t-tau), X_1(t); X_2(t-tau), X_2(t))
%              'MI1': Multi (Mutual) information. I(X_1(t); X_2(t)). (IIT1.0)
%           options.type_of_MIPsearch
%              'Exhaustive': exhaustive search
%              'Queyranne': Queyranne algorithm
%              'REMCMC': Replica Exchange Monte Carlo Method (Not available now.)
%           options.normalization (available only for Gaussian dist.)
%              0: without normalization of phi by Entropy (default)
%              1: with normalization of phi by Entropy
%
%
% OUTPUT:
%           Z_MIP: the MIP
%           phi_MIP: the amount of integrated information at the MIP
%
%           When options.type_of_MIPsearch is 'Exhaustive', the following
%           variables are also available.
%                Zs: all the bi-partitions. 2^(N-1)-1 by N matrix, where N 
%                    is the number of elemenets, and 2^(N-1)-1 is the
%                    number of bi-partitions.
%                phis: the amount of integrated information at respective partitions. 2^(N-1)-1 by 1 vector.
% 
%
% Jun Kitazono & Masafumi Oizumi, 2018


switch options.type_of_MIPsearch
    case 'Exhaustive'
        [Z_MIP, phi_MIP, Zs, phis] = MIP_Exhaustive( probs, options );
    case 'Queyranne'
        [Z_MIP, phi_MIP] = MIP_Queyranne( probs, options );
%     case 'REMCMC'
%          [Z_MIP, phi_MIP, ...
%     phi_history, State_history, Exchange_history, T_history, wasConverged, NumCalls] = MIP_REMCMC( probs, options);
end