function [minus_I_s, minus_I_s_d] = I_s_I_s_d(beta,C_D_beta1_inv,Cov_X_inv,Cov_X,Cov_Y,C_D_cond,S_left,S_right,I_s_d_Const_part)
    C_D_beta_inv = beta*C_D_beta1_inv; % 2nd term of eq. (26)
    Q_inv = Cov_X_inv + C_D_beta_inv; % Q_inv
    
    norm_t = 1/2*logdet(Q_inv) + 1/2*logdet(Cov_X);
    R = beta*inv(C_D_cond) - beta^2*S_left/Q_inv*S_right;
    
    trace_t = 1/2*trace(Cov_Y*R);
    I_s = norm_t + trace_t - beta*I_s_d_Const_part;
    minus_I_s = -I_s;
    
    Q_d = -Q_inv\C_D_beta1_inv/Q_inv; %derivative of Q
    R_d = inv(C_D_cond) - beta*S_left*2/Q_inv*S_right - beta*S_left*beta*Q_d*S_right;
    
    I_s_d = 1/2*(-trace(Q_inv*Q_d) + trace(Cov_Y*R_d)) - I_s_d_Const_part; % derivative of I_s
    minus_I_s_d = -I_s_d;
end