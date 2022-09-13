function SI = SI_dis(probs, q_probs)

%-------------------------------------------------------------------------------------------------
% PURPOSE: calculate stochastic interaction in discretized data
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
%   SI: stochastic interaction proposed by Ay or Barrett & Seth  
%-------------------------------------------------------------------------------------------------
%
% Masafumi Oizumi, 2018

p_past = probs.past;
p_joint = probs.joint;
q_TPM = q_probs.TPM;

TNS = length(p_past);

H_cond = 0;
H_cond_q = 0;

% i: present, j: past
for i=1: TNS
    for j=1: TNS
        if p_joint(i,j) ~= 0
            H_cond = H_cond - p_joint(i,j)*log(p_joint(i,j)/p_past(j));
        end
        
        if q_TPM(i,j) ~= 0
            H_cond_q = H_cond_q - p_joint(i,j)*log(q_TPM(i,j));
        end
    end
end

SI = H_cond_q - H_cond;

end