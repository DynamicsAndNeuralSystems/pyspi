function H = H_dis(p)

%% compute entropy in discretized data
p(p==0) = 1;
H = - sum(p.*log(p));