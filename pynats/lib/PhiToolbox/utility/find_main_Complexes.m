function [main_complexes, main_phis] = find_main_Complexes(complexes, phis)

nSubsets = length(complexes);
isMainComplex = true(nSubsets, 1);

for i=1: nSubsets
    cand = complexes{i};
    phi = phis(i);
    for j=1: nSubsets
        tmp = complexes{j};
        if isequal(tmp, intersect(cand,tmp))
            if phi < phis(j)
                isMainComplex(i) = false;
%                 disp(cand)
%                 disp(tmp)
%                 fprintf('i=%d cand_phi=%f tmp_phi=%f\n',i,phi,phis(j));
                break;
            end
        end
    end
end

complex_index = isMainComplex == true;
main_complexes = complexes(complex_index);
main_phis = phis(complex_index);

end