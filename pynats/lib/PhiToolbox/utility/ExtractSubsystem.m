function probs_Sub = ExtractSubsystem( type_of_dist, probs, indices )
% ExtractSubsystem
% 

N_sub = length(indices);
probs_Sub.number_of_elements = N_sub;
N = probs.number_of_elements;

switch type_of_dist
    case 'Gauss'
        probs_Sub.Cov_X = probs.Cov_X(indices, indices);
        if isfield(probs,'Cov_XY')
            probs_Sub.Cov_XY = probs.Cov_XY(indices, indices);
            probs_Sub.Cov_Y = probs.Cov_Y(indices, indices);
        end
    case 'discrete'
        N_st = probs.number_of_states;
        probs_Sub.number_of_states = N_st;
        %% prepare tables
        sigma_table = zeros(N,N_st^N);
        for i=1: N_st^N
            sigma = base10toM(i-1,N,N_st);
            sigma_table(:,i) = sigma;
        end
        
        i_vec = 0: N_sub-1;
        pow_vec = N_st.^i_vec;
        %%
        
        if isfield(probs,'past')
            probs_Sub.past = marginalize_table(probs.past, indices, N_st, sigma_table, pow_vec);
            probs_Sub.joint = marginalize_table(probs.joint, indices, N_st, sigma_table, pow_vec);
            probs_Sub.present = marginalize_table(probs.present, indices, N_st, sigma_table, pow_vec);
        end
        if isfield(probs,'p')
            probs_Sub.p = marginalize_table(probs.p, indices, N_st, sigma_table, pow_vec);
        end
    case 'UndirectedGraph'
        probs_Sub.g = probs.g(indices, indices);
        
end


end