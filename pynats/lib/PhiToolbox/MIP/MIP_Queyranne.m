function [Z_MIP, phi_MIP] = MIP_Queyranne( probs, options)
% Find the minimum information partition (MIP) using Queyranne's
% algorithm
%
% INPUTS:   
%   see MIP_search_probs
%
% OUTPUT:
%      Z_MIP: the esetimated MIP
%      phi_MIP: amount of integrated information at the estimated MIP

N = probs.number_of_elements;
type_of_dist = options.type_of_dist;
type_of_phi = options.type_of_phi;

isNormalization = false;
if isfield(options, 'normalization')
    if options.normalization == 1
        isNormalization = true;
    end
end
if strcmpi(type_of_phi, 'MI1') && strcmpi(type_of_dist, 'Gauss') && ~isNormalization
    [IndexOutput] = QueyranneAlgorithmNormal(probs.Cov_X, 1:N);
    
    phi_MIP = MI1_Gauss(probs.Cov_X, indices2bipartition(IndexOutput, N));
else
    F = @(indices)phi_comp_probs(type_of_dist, type_of_phi, indices2bipartition(indices, N), probs, options);

    [IndexOutput] = QueyranneAlgorithm( F, 1:N );
    phi_MIP = F(IndexOutput);
end

if ismember(1, IndexOutput)
    Z_MIP = 2*ones(1,N);
    Z_MIP(IndexOutput) = 1;
else
    Z_MIP = ones(1, N);
    Z_MIP(IndexOutput) = 2;
end

end