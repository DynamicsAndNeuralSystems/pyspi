function weights = EachMaxPhi( indices, phis, nelems, numTops )
% Take maximum phi of each element. 
%
% INPUT:
%    indices: The indices of subsets. (#subsets by 1 cell array. Each cell contains indices of a subset.)
%    phis: The amount of integrated information for subsets. (#subsets by 1
%    vector)
%    nelems: The number of elements in the entire system.
%    numTops: Only the subsets that have Top (numTops) phi are taken into
%    consideration. 
%
% OUTPUT:
%    weights: if an element is included at least one of the subsets with
%    Top (numTops) phi, its maximum amount of phi among Top (NumTops) is
%    stored. 
%
% Jun Kitazono, 2019

[phis_sort, index_phis_sort] = sort(phis, 'descend');

weights = zeros(1, nelems);
for i = numTops: -1: 1
    weights(1, indices{index_phis_sort(i)}) = phis_sort(i);
end

end