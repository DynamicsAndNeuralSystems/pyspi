function P = power_set( S )
    n = numel(S);
    x = 1:n;
    P = cell(1,2^n-1);
    p_ix = 1;
    for nn = 1:n
        a = combnk(x,nn);
        for j=1:size(a,1)
            P{p_ix} = S(a(j,:));
            p_ix = p_ix + 1;
        end
    end    
end