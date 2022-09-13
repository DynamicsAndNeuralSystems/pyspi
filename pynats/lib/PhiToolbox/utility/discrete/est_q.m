function q_probs = est_q(Z, N_st, probs)

%-------------------------------------------------------------------------------------------------
% PURPOSE: estimate mismatched probability distribution q from the original probability distribution p 
%
% INPUTS:
%   p_past: probability distribution of past state (X^t-tau)
%   joint: joint distribution of X^t (present) and X^(t-\tau) (past)
%   p_present: probability distribution of present state (X^t)
%   N_st: number of states
%   Z: partition with which integrated information is computed
%
% OUTPUTS:
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

N = length(Z); % number of units

N_c = max(Z); % number of clusters
M_cell = cell(N_c,1);
for k=1: N_c
    M_cell{k} = find(Z==k);
end

%% prepare tables

sigma_table = zeros(N,N_st^N);
for i=1: N_st^N
    sigma = base10toM(i-1,N,N_st);
    sigma_table(:,i) = sigma;
end

pow_cell = cell(N,1);
for i=1: N
    i_vec = 0: i-1;
    pow_vec = N_st.^i_vec;    
    pow_cell{i} = pow_vec; 
end

%%

if isfield(probs,'joint')
    q_past = cell(N_c,1);
    q_joint = cell(N_c,1);
    q_present = cell(N_c,1);
else
    q_cell = cell(N_c,1);
end

for k=1: N_c
    index = M_cell{k};
    N_sub = length(index);
    pow_vec = pow_cell{N_sub};
    
    if isfield(probs,'joint')
        q_past{k} = marginalize_table(probs.past, index,N_st,sigma_table, pow_vec);
        q_joint{k} = marginalize_table(probs.joint, index,N_st,sigma_table, pow_vec);
        q_present{k} = marginalize_table(probs.present, index,N_st,sigma_table, pow_vec);
    else
        q_cell{k} = marginalize_table(probs.p, index,N_st,sigma_table, pow_vec);
    end
end



if isfield(probs,'joint')
    q_ind = cell(N_c,1);
    for k=1: N_c
        N_k = length(M_cell{k}); % number of units in each cluster
        q_k = zeros(N_st^N_k,N_st^N_k);
        for i=1: N_st^N_k % present
            for j=1: N_st^N_k % past
                if q_joint{k}(i,j) ~= 0
                    q_k(i,j) = q_joint{k}(i,j)/q_past{k}(j);
                end
            end
        end
        q_ind{k} = q_k;
    end
    
    q_TPM = ones(N_st^N,N_st^N);
    
    if sum(abs(Z-(1:N))) == 0
        for i=1: N_st^N
            i_b = sigma_table(:,i) + 1;
            for j=1: N_st^N
                j_b = sigma_table(:,j) + 1;
                for k=1: N
                    q_TPM(i,j) = q_TPM(i,j)*q_ind{k}(i_b(k),j_b(k));
                end
            end
        end
    else
        
        for i=1: N_st^N
            i_b = convert_index(i,M_cell,N_c,sigma_table, pow_cell);
            for j=1: N_st^N
                j_b = convert_index(j,M_cell,N_c,sigma_table, pow_cell);
                for k=1: N_c
                    q_TPM(i,j) = q_TPM(i,j)*q_ind{k}(i_b(k),j_b(k));
                end
            end
        end
        
    end
    
end

if isfield(probs,'joint')
    q_probs.TPM = q_TPM;
    q_probs.past = q_past;
    q_probs.joint = q_joint;
    q_probs.present = q_present;
else
    q_probs.q = q_cell;
end

end

function x_local = convert_index(x_global,M_cell, N_c, sigma_table, pow_cell)

x_gb = sigma_table(:,x_global); % covert N_bin base
x_local = zeros(N_c,1);

for k=1: N_c
    M = M_cell{k};
    x_lb = x_gb(M); % N_st base index in each cluster
    
    N_sub = length(M);
    pow_vec = pow_cell{N_sub};
    
    x_l = baseMto10_table(x_lb,pow_vec) + 1; % index in each cluster
    x_local(k) = x_l;
end

end