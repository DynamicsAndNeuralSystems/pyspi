function MI1 = MI1_dis(probs, q_probs)

%-------------------------------------------------------------------------------------------------
% PURPOSE: calculate mutual (multi) information between subsystems in discretized data
%
% INPUTS:
%   probs: probability distributions computed from X
%       probs.past: probability distribution of past state (X^t-tau)
%       probs.joint: joint distribution of X^t (present) and X^(t-\tau) (past)
%       probs.present: probability distribution of present state (X^t)
%      
%       probs.p: probability distribution of X only used for mutual
%       information (MI)
%   q_probs: mismatched probability distributions q
%       q_probs.TPM: mismatched conditional probability distribution of present state
%                              given past state (q(X^t|X^t-tau))
%       q_probs.past: mismatached probability distribution of past state (X^t-tau)
%       q_probs.joint: mismatched joint distribution of X^t (present) and X^(t-\tau) (past)
%       q_probs.present: mismatched probability distribution of present state (X^t)
%       
%       q_probs.q: mismtached probability distribution of X only used for mutual
%       information (MI)
%
% OUTPUTS:
%   MI1: mutual information between subsystems 
%-------------------------------------------------------------------------------------------------
%
% Masafumi Oizumi, 2018

p = probs.p;
q = q_probs.q;

N_c = length(q);
H_part = zeros(N_c,1);
for i=1: N_c
    H_part(i) = H_dis(q{i});
end
H_joint = H_dis(p);

MI1 = sum(H_part) - H_joint;

end