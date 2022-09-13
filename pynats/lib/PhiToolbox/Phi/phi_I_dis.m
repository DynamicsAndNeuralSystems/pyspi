function phi_I = phi_I_dis(probs,q_probs)

%-------------------------------------------------------------------------------------------------
% PURPOSE: calculate mutual information based integrated information proposed in discretized data
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
%   phi_I: mutual information based integrated information proposed by  Barrett & Seth  
%-------------------------------------------------------------------------------------------------
%
% Masafumi Oizumi, 2018

p_past = probs.past;
p_joint = probs.joint;
p_present = probs.present;

q_past = q_probs.past;
q_joint = q_probs.joint;
q_present = q_probs.present;

N_c = length(q_past);
I_vec = zeros(N_c,1);

I = I_dis(p_past,p_joint,p_present);
 
for i=1: N_c
 I_vec(i) = I_dis(q_past{i},q_joint{i},q_present{i});
end

phi_I = I - sum(I_vec);