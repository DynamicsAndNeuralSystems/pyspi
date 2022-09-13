function [complexes, phis_complexes, main_complexes, phis_main_complexes, Res] = Complex_Recursive( probs, options )
% Find complexes using the recursive MIP algorithm
%
% INPUTS:
%           probs: probability distributions for computing phi
%           
%           In the Gaussian case
%               When options.type_of_phi is 'MI1'
%                  probs.Cov_X: covariance of data X
%               When options.type_of_phi is NOT 'MI1'
%                  probs.Cov_X: covariance of data X (past, t-tau)
%                  probs.Cov_XY: cross-covariance of X (past, t-tau) and Y (present, t)
%                  probs.Cov_Y: covariance of data Y (present, t)
%           In the discrete case
%               When options.type_of_phi is 'MI1'
%                  probs.p: probability distribution of X
%               When options.type_of_phi is NOT 'MI1'
%                  probs.past: probability distribution of past state (X(t-tau))
%                  probs.joint: joint distribution of X(t) (present) and X(t-tau) (past)
%                  probs.present: probability distribution of present state X(t-tau)
%
%           options: options for computing phi and for MIP search
%           
%           options.type_of_dist:
%              'Gauss': Gaussian distribution
%              'discrete': discrete probability distribution
%           options.type_of_phi:
%              'SI': phi_H, stochastic interaction
%              'Geo': phi_G, information geometry version
%              'star': phi_star, based on mismatched decoding
%              'MI': Multi (Mutual) information, I(X_1(t-tau), X_1(t); X_2(t-tau), X_2(t))
%              'MI1': Multi (Mutual) information. I(X_1(t); X_2(t)). (IIT1.0)
%           options.type_of_MIPsearch
%              'Exhaustive': exhaustive search
%              'Queyranne': Queyranne algorithm
%           options.groups: search complex based on predefined groups. 
%                           A group is considered as an element, which is
%                           not subdivided further. 
%
% OUTPUTS:
%    complexes: indices of elements in complexes
%    phis_complexes: phi statisticd with the MIP of complexes
%    main_complexes: indices of elements in main complexes
%    phis_main_complexes: phi statisticd with the MIP of main complexes
%   
%    Res.Z: Each row of Z represents a complex candidate. This is a subset
%           belonging to \mathcal{V} in Kitazono et al., 2020. The i-th
%           element of a row has a nonzero value (1 or 2) when the element
%           is included in the candidate and zero when the element is NOT
%           included. The value 1 and 2 indicate the MIP of a candidate. 
%           Ex.: Res.Z(i,:): [0, 0, 2, 1, 2, 0, 2]
%                             1  2  3  4  5  6  7
%                (The 3rd, 4th, 5th, and 7th elements are included in the
%                i-th candidate. Its MIP divides it into {4} and {3,5,7}.) 
%           
%    Res.phi: phi statisticd with the MIPs of the complex candidates. The
%             i-th element of Res.phi is phi of i-th candidate statisticd
%             with its MIP. 
%
%    Res.parent: The parent of each complex candidate. When the i-th
%                element of Res.parent is j (i<j), this means that the j-th
%                candidate is the parent of i-th candidate in the
%                hierarchy. When i-th element is zero, this means the i-th
%                candidate is the root node (the whole system). 
%                
%    Res.isComplex: This indicates whether each candidate is a complex or
%                   not (1: a complex, 0: not a complex). 
%    Res.isMainComplex: This indicates whether each candidate is a main
%                       complex or not (1: a main complex, 0: not a main complex). 
%
% Jun Kitazono, 2020

Res = Complex_RecursiveFunction( probs, options );
[complexes, phis_complexes, isComplex, ...
    main_complexes, phis_main_complexes, isMainComplex] = find_Complexes_fromRes(Res);
Res.isComplex = isComplex;
Res.isMainComplex = isMainComplex;

end

function Res = Complex_RecursiveFunction( probs, options )

N = probs.number_of_elements;

type_of_MIPsearch = options.type_of_MIPsearch;
type_of_dist = options.type_of_dist;

if N < 2
    Res.Z = [];
    Res.phi = [];
    Res.parent = 0;
else
    
    switch type_of_MIPsearch
        case 'Exhaustive'
            [Z_MIP, phi_MIP] = MIP_Exhaustive( probs, options );
        case 'Queyranne'
            [Z_MIP, phi_MIP] = MIP_Queyranne( probs, options );
    end
    
    indices_L = find(Z_MIP==1);
    indices_R = find(Z_MIP==2);
    
    probs_L = ExtractSubsystem( type_of_dist, probs, indices_L );
    probs_R = ExtractSubsystem( type_of_dist, probs, indices_R );

    Res_L = Complex_RecursiveFunction( probs_L, options );
    Res_R = Complex_RecursiveFunction( probs_R, options );
    
    Res = Concatenate_Res(N, Z_MIP, phi_MIP, indices_L, indices_R, Res_L, Res_R);
    
end

end

function Res = Concatenate_Res(N, Z, phi, indices_L, indices_R, Res_L, Res_R)

[nRows_L, ~] = size(Res_L.Z);
[nRows_R, ~] = size(Res_R.Z);
Res.Z = zeros(nRows_L+nRows_R+1, N);
Res.Z(1:nRows_R, indices_R) = Res_R.Z;
Res.Z((nRows_R+1):(nRows_R+nRows_L), indices_L) = Res_L.Z;
Res.Z(nRows_R+nRows_L+1,:) = Z;

Res.phi = [Res_R.phi; Res_L.phi; phi];

Res.parent = zeros(nRows_L+nRows_R+1, 1);
Res.parent(1:nRows_R, 1) = Res_R.parent;
Res.parent((nRows_R+1):(nRows_R+nRows_L),1) = nRows_R + Res_L.parent;

if nRows_R > 0
    Res.parent(nRows_R, 1) = nRows_L+nRows_R+1;
end
if nRows_R+nRows_L>0
    Res.parent(nRows_R+nRows_L, 1) = nRows_L+nRows_R+1;
end

end

function [complexes, phis_complexes, isComplex, main_complexes, phis_main_complexes, isMainComplex] = find_Complexes_fromRes(Res)

nSubsets = length(Res.phi);
phi_temp_max = zeros(nSubsets, 1);
isComplex = false(nSubsets, 1);
isMainComplex = false(nSubsets, 1);
mrac = zeros(nSubsets, 1);

phi_temp_max(nSubsets) = Res.phi(nSubsets);
isComplex(nSubsets) = true;
isMainComplex(nSubsets) = true;
mrac(nSubsets) = nSubsets;
for i = nSubsets-1: -1 :1
    if Res.phi(i) > phi_temp_max(Res.parent(i))
        phi_temp_max(i) = Res.phi(i);
        isComplex(i) = true;
        
        isMainComplex( mrac(Res.parent(i)) ) = false;
        isMainComplex(i) = true;
        mrac(i) = i;
    else
        phi_temp_max(i) = phi_temp_max(Res.parent(i));
        
        mrac(i) = mrac(Res.parent(i));
    end
end

[complexes, phis_complexes] = sort_Complexes(isComplex, Res);
[main_complexes, phis_main_complexes] = sort_Complexes(isMainComplex, Res);

end

function [complexes, phis] = sort_Complexes(isComplex, Res)

nComplexes = nnz(isComplex);
complexes = cell(nComplexes, 1);
Zs = Res.Z(isComplex,:);
for i = 1:nComplexes
    complexes{i} = find(Zs(i,:));
end

phis = Res.phi(isComplex);

[phis_sorted, idx_phis_sorted] = sort(phis, 'descend');
phis = phis_sorted;
complexes = complexes(idx_phis_sorted);

end