function [phi_star, I, H] = phi_star_dis(probs, q_probs)

%-------------------------------------------------------------------------------------------------
% PURPOSE: calculate integrated information "phi_star" in discretized
% data. See Oizumi et al., 2016, PLoS Comp for the details of phi_star. The
% equation number in the codes refers to Oizumi et al., 2016, PLoS Comp.
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
%-------------------------------------------------------------------------------------------------
%
% Masafumi Oizumi, 2018

p_past = probs.past;
p_joint = probs.joint;
p_present = probs.present;
q_TPM = q_probs.TPM;

TNS = length(p_past); % total number of all possible states

%% find beta by a quasi-Newton method
beta =  1; % initial value

% set options of minFunc
Options.Method = 'lbfgs';
Options.Display = 'off';
% Options.useMex = 0;

%% minimize  negative I_s
[beta,minus_I_s,~,~] = minFunc(@I_s_I_s_d,beta,Options);
I_s = -minus_I_s;
[I, H, H_cond] = I_dis(p_past,p_joint,p_present);
phi_star = I - I_s;

% fprintf('beta=%f phi_star=%f I=%f H=%f\n',beta, phi_star, I, H);

    function [minus_I_s, minus_I_s_d] = I_s_I_s_d(beta)
        if beta < 10^-20
            beta = 10^-20;
        end
        
        I_s1 = 0; % the first term in Eq. (20)
        I_s2 = 0; % the second term in Eq. (20)
        
        % i: present state, j: past state
        for i=1: TNS
            Den = 0;
            for j=1: TNS
                if q_TPM(i,j) ~= 0
                    I_s2 = I_s2 + beta*p_joint(i,j)*log(q_TPM(i,j));
                end
                Den = Den + p_past(j)*q_TPM(i,j)^beta;
            end
            if Den ~= 0
                I_s1 = I_s1 - p_present(i)*log(Den);
            end
        end
        I_s = I_s1 + I_s2; % Eq. (20)
        minus_I_s = - I_s;
        
        I_s1_d = 0; % the derivative of I_s1
        I_s2_d = 0; % the derivative of I_s2
        
        % i: present state, j: past state
        for i=1: TNS
            Num = 0;
            Den = 0;
            for j=1: TNS
                if q_TPM(i,j) ~= 0
                    I_s2_d = I_s2_d + p_joint(i,j)*log(q_TPM(i,j));
                    Num = Num + p_past(j)*q_TPM(i,j)^beta*log(q_TPM(i,j));
                end
                Den = Den + p_past(j)*q_TPM(i,j)^beta;
            end
            
            if Num ~= 0
                I_s1_d = I_s1_d - p_present(i)*Num/Den;
            end
        end
        
        I_s_d = I_s1_d + I_s2_d;
        minus_I_s_d = - I_s_d;
    end

end