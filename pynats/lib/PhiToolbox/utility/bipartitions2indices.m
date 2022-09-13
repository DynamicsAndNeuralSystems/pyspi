function indices = bipartitions2indices( Zs )
%
% 

nRows_Zs = size(Zs, 1);
indices = cell(nRows_Zs, 1);

for i = 1:nRows_Zs
    indices{i} = find(Zs(i,:));
end

end