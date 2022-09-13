function p_m = marginalize_table(p,index,N_st,sigma_table,pow_vec)
% ---------------------------------
% PURPOSE: marginalize probability distribution p(X)
% Input
% p: probability distribution of X
% index: marginalized out other variables than those indicated by index
%
% Output
% p_m: marginalized probability distribution

%----------------------------------

TNS = length(p);
N_sub = length(index);

if size(p,1) == 1 || size(p,2) == 1
    p_m = zeros(N_st^N_sub,1);
    for i=1: TNS
        sigma = sigma_table(:,i);
        % marginalize
        sub_i = baseMto10_table(sigma(index), pow_vec) + 1;
        p_m(sub_i) = p_m(sub_i) + p(i);
    end
else
    p_m = zeros(N_st^N_sub,N_st^N_sub);
    
    for i=1: TNS
        sigma_i = sigma_table(:,i);
        sub_i = baseMto10_table(sigma_i(index),pow_vec)+1;
        for j=1: TNS
            sigma_j = sigma_table(:,j);
            sub_j = baseMto10_table(sigma_j(index),pow_vec)+1;
            p_m(sub_i,sub_j) = p_m(sub_i,sub_j) + p(i,j);
        end
    end
end