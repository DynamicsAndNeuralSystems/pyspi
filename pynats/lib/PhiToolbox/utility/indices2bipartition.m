function Z = indices2bipartition( indices, N )
% Transform indices to a bi-partition
% Ex. N=5, indices=[1,2,4] -> Z=[1,1,2,1,2];

if iscell(indices)
    ind = cell2mat(indices);
else
    ind = indices;
end

Z = 2*ones(1, N);
Z(ind) = 1;

end