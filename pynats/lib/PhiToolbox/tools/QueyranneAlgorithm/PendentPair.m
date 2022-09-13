function [ pendantPair ] = PendentPair( F, index, ind )
%% [ pendantPair ] = PendentPair( F, index, ind ) 
%% 
%% This function is the core sub-routine of the QueyranneAlgorithm.m
%% 
%% First written by Shohei Hidaka   Oct 22nd, 2015.
%% Revised (added its description in comments) by Shohei Hidaka Jan 13th, 2016.
%%
%% See also: QueyranneAlgorithm
%%
%% References: Queyranne, M. (1998). Minimizing symmetric submodular functions. Mathematical Programming, 82(1-2), 3-12.
%% http://link.springer.com/article/10.1007%2FBF01585863
if nargin < 2
    ind = 1;
end

for i = 1:( length( index ) - 1 )
%     disp(ind)
    indc = setdiff( 1:length( index ), ind );
    candidates = index( indc );
    
    TCtemp = zeros( 1, length( candidates ) );
    for j = 1:length( candidates )
        TCtemp( j ) = F( [ index( ind ), candidates( j ) ] ) - F( [ candidates( j ) ] );
    end
    [ TCgreedy, minind ] = min( TCtemp );
    ind = [ ind, indc( minind ) ];
end
pendantPair = ind;

