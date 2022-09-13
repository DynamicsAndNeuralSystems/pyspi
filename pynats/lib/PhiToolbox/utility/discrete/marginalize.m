function p_m = marginalize(p,index,N,N_st)
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
        sigma = base10toM(i-1,N,N_st);
        % marginalize
        sub_i = baseMto10(sigma(index),N_st)+1;
        p_m(sub_i) = p_m(sub_i) + p(i);
    end
else
    p_m = zeros(N_st^N_sub,N_st^N_sub);
    
    for i=1: TNS
        sigma_i = base10toM(i-1,N,N_st);
        sub_i = baseMto10(sigma_i(index),N_st)+1;
        for j=1: TNS
            sigma_j = base10toM(j-1,N,N_st);
            sub_j = baseMto10(sigma_j(index),N_st)+1;
            p_m(sub_i,sub_j) = p_m(sub_i,sub_j) + p(i,j);
        end
    end
end