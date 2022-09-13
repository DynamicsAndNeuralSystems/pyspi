function [Z_MIP, phi_MIP, Zs, phis] = MIP_Exhaustive( probs, options)
% Find the Minimum Informamtion Partition using the exhaustive search
% from probability distirubtions
%
% INPUTS: 
%   see MIP_search_probs
% OUTPUT:
%     Z_MIP: the MIP
%     phi_MIP: the amount of integrated information at the MIP
%     Zs: all the bi-partitions. 2^(N-1)-1 by N matrix, where N is the number of elemenets.
%     phis: the amount of integrated information at respective partitions. 2^(N-1)-1 by 1 vector.
%           
% 
% Jun Kitazono & Masafumi Oizumi, 2018

N = probs.number_of_elements;
type_of_dist = options.type_of_dist;
type_of_phi = options.type_of_phi;

all_comb = power_set(2:N);
nComb = length(all_comb);
phis = zeros(nComb,1);
Zs = zeros(nComb,N);


parfor i=1: nComb
    subcluster = all_comb{i};
    Z = ones(1,N);
    Z(subcluster) = 2;
    % compute phi
    phis(i) = phi_comp_probs(type_of_dist, type_of_phi, Z, probs, options);
    Zs(i,:) = Z;
end

[phi_MIP, MIP_ind] = min(phis);

Z_MIP = ones(1,N);
Z_MIP(all_comb{MIP_ind}) = 2;



end