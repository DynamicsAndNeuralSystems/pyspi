function [ IndexOutput ] = QueyranneAlgorithm( F, index )
% [ IndexOutput ] = QueyranneAlgorithm( F, Index ) 
% 
% This function is an implementation of the Queyranne's algorithm for a
% symmetric submodular function F of the set Index.
%
% Input ---
%       F: A function hundle, F( X ) is supposed a symmetric submodular function of X.
%       Index: A set of indices implying the domain of the F.
%
% Output ---
%       IndexOutput: the index set minimizing the function F.
% 
% First written by Shohei Hidaka   Oct 22nd, 2015.
% Revised (added its description in comments) by Shohei Hidaka Jan 13th, 2016.
% Revised (rename PendentPair and add one line for the preprocess of the index ) by Shohei Hidaka March 18th, 2016.
% Revised (additional minor function to sort the output argument) by Shohei Hidaka on April 5th, 2016
% Revised (additional minor part for input and output argument) by Shohei Hidaka on April 26th, 2016
%
% See also: MaxTotalCorr, PendentPair
%
% References: Queyranne, M. (1998). Minimizing symmetric submodular functions. Mathematical Programming, 82(1-2), 3-12.
% http://link.springer.com/article/10.1007%2FBF01585863

%% 
assert( isa( F, 'function_handle' ) )
assert( isa( index, 'cell' )||isa( index, 'double' ) )

%% Added March 18th, 2016./ further revised on April 26th.
if ~iscell( index )
    index = arrayfun( @(x) x, 1:length( index ), 'UniformOutput', false );
end
%% Added March 18th, 2016.

N = length( index );
M = sum( cellfun( @numel, index ) ); % further revised on April 26th.

indexrec = cell( 1, N - 1 );
f = zeros( 1, N - 1 );
for i = 1:( N - 1 )
    %fprintf( 'i = %d: ', i );
    [ pp ] = PendentPair( F, index, 1 );
    indexrec( i ) = index( pp( end ) );
    f( i ) = F( index( pp( end ) ) ); %TotalCorr( index( pp( end ) ), CV );
    index = index( pp );
    index = [ index( 1:end-2 ), { cell2mat( index( end-1:end ) ) } ];
    %fprintf( 'F = %.3f\n', f( i ) );
end
[ tmp, minind ] = min( f );
%% Revised on April 5th, 2016
%IndexOutput = indexrec{ minind };
% if numel( indexrec{ minind } ) > N/2
%     IndexOutput = setdiff( 1:N, indexrec{ minind } );
% else
%     IndexOutput = sort( indexrec{ minind } );
% end
% further revised on April 26th.
if ~isempty( f ) & numel( indexrec{ minind } ) > M/2
    IndexOutput = setdiff( cell2mat( cellfun( @(x) x(:)', index, 'UniformOutput', false ) ), indexrec{ minind } );
elseif ~isempty( f )
    IndexOutput = sort( indexrec{ minind } );
end





