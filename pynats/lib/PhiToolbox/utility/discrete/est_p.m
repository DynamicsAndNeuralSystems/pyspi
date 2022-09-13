function probs = est_p(X, N_s, tau, isjoint)

%-------------------------------------------------------------------------------------------------
% PURPOSE: estimate probability distributions of past (p_past), present (p_present) and 
% joint distribution of past and present (joint) from discretized time series data X 
%
% INPUTS:
%   X: discretized time series data in form units x time
%   N_s: the number of states in each unit
%   tau: time lag between past and present
%   isjoint:  whether or not joint probability distributions are computed
%
% OUTPUTS:
%   probs: probability distributions computed from X
%       probs.past: probability distribution of past state (X^t-tau)
%       probs.joint: joint distribution of X^t (present) and X^(t-\tau) (past)
%       probs.present: probability distribution of present state (X^t)
%      
%       probs.p: probability distribution of X only used for mutual
%       information (MI)
%
%-------------------------------------------------------------------------------------------------
%
% Masafumi Oizumi, 2018

%%
if nargin < 4
    isjoint = 1;
end

N = size(X,1);
T = size(X,2);

i_vec = 0: N-1;
pow_vec = N_s.^i_vec;

if isjoint
    p_past = zeros(N_s^N,1);
    p_present = zeros(N_s^N,1);
    p_joint = zeros(N_s^N,N_s^N);
    
    for t=1: T-tau
        past_s = baseMto10_table(X(:,t) - 1, pow_vec) + 1; % past state
        p_past(past_s) = p_past(past_s) + 1;
        present_s = baseMto10_table(X(:,t+tau) - 1, pow_vec) + 1; % present state
        p_present(present_s) = p_present(present_s) + 1;
        p_joint(present_s,past_s) = p_joint(present_s,past_s) + 1;
    end
    
    p_past = p_past/(T-tau);
    p_present = p_present/(T-tau);
    p_joint = p_joint/(T-tau);
    
    probs.past = p_past;
    probs.present = p_present;
    probs.joint = p_joint;
else
    p = zeros(N_s^N,1);
    
    for t=1: T
        s = baseMto10_table(X(:,t) - 1, pow_vec) + 1; % past state
        p(s) = p(s) + 1;
    end
    
    p = p/T;
    probs.p = p;
end

% probs.number_of_elements = size(X, 1);
% probs.number_of_states = number_of_states;