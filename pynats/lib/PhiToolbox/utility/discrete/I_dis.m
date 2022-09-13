function [I, H, H_cond] = I_dis(p_past,p_joint,p_present)

%-------------------------------------------------------------------------------------------------
% PURPOSE: calculate mutual information
%
% INPUTS:
%   p_past: probability distribution of past state (X^t-tau)
%   joint: joint distribution of X^t (present) and X^(t-\tau) (past)
%   p_present: probability distribution of present state (X^t)
%
% OUTPUTS:
%   phi_star: integrated information based on mismatched decoding (Oizumi et al., 2016, PLoS Comp)
%   I: mutual information between X^t and X^(t-\tau)
%   H: entropy of X^t
%-------------------------------------------------------------------------------------------------
%
% Masafumi Oizumi, 2018

if nargin < 3
    p_present = p_past;
end

p_present(p_present==0) = 1;
H = - sum(p_present.*log(p_present));
TNS = length(p_present); % total number of all possible states

H_cond = 0; % conditional entropy
% i: present, j: past
for i=1: TNS
    for j=1: TNS
        if p_joint(i,j) ~= 0
            H_cond = H_cond - p_joint(i,j)*log(p_joint(i,j)/p_past(j));
        end
    end
end

I = H - H_cond;