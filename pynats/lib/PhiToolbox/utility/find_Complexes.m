function [complexes, phis_complexes] = find_Complexes( indices, phis, flag_sorted_descending )
% Find main complexes (IIT2.0) from the results of exhaustive evaluation of phi
%
% INPUTS:   
%           indices: indices of every subsystem
%           phis: phi for every subsystem
%           flag_sorted_descending: indicator whether or not values are sorted in descending order. 
%
% OUTPUT:
%           complexes: indices of complexes
%           phis_complexes: phi of complexes

if nargin < 3
    flag_sorted_descending = 0;
end

if flag_sorted_descending == 0
    [phis_sorted_descending, sort_index] = sort(phis, 'descend');
    indices_sorted_descending = indices(sort_index);
end

isC = true(length(phis_sorted_descending), 1);


for i = length(phis_sorted_descending):-1:1
    A = indices_sorted_descending{i};
    for j = (i-1):-1:1
        B = indices_sorted_descending{j};
        
        AminusB = setdiff(A, B);
        %BminusA = setdiff(B, A);
        
        isempty_AminusB = isempty(AminusB); % Is A included in B?
        %isempty_BminusA = isempty(BminusA); % Is B included in A?
        
        if isempty_AminusB% || isempty_BminusA
            isC(i) = false;
            break;
        end
    end
end

complexes = indices_sorted_descending(isC);
phis_complexes = phis_sorted_descending(isC);


end

