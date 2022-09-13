function [ IndexOutput ] = QueyranneAlgorithmNormal( CovMat, index, init_ind )
% [ IndexOutput ] = QueyranneAlgorithmNormalModified( CovMat, index, init_ind ) 
% 
% This function is an implementation of the Queyranne's algorithm for a
% symmetric submodular function F of the set Index.
%
% Input ---
%       CovMat: A covariance matrix.
%       index: A set of indices implying the domain of the F.
%
% Output ---
%       IndexOutput: the index set minimizing the function F.
% 
% First written by Shohei Hidaka   Oct 22nd, 2015.
% Revised (added its description in comments) by Shohei Hidaka Jan 13th, 2016.
% Revised (rename PendentPair and add one line for the preprocess of the index ) by Shohei Hidaka March 18th, 2016.
% Rebuilt (specialized implementation for the normal distribution) by Shohei Hidaka on April 14th.
% Revised (additional minor part for input and output argument) by Shohei Hidaka on April 26th, 2016
%
% See also: MaxTotalCorr, PendentPair
%
% References: Queyranne, M. (1998). Minimizing symmetric submodular functions. Mathematical Programming, 82(1-2), 3-12.
% http://link.springer.com/article/10.1007%2FBF01585863

if nargin < 3
    init_ind = 1;
end
%% Added March 18th, 2016./ further revised on April 26th.
if ~iscell( index )
    index = arrayfun( @(x) x, 1:length( index ), 'UniformOutput', false );
end
%% Added March 18th, 2016.

F = @( ind ) TotalCorrDiff( ind, CovMat );
N = length( index );
M = sum( cellfun( @numel, index ) ); % further revised on April 26th.

indexrec = cell( 1, N - 1 );
f = zeros( 1, N - 1 );
for i = 1:( N - 1 );
    % fprintf( 'i = %d: ', i );
    if i == 1
        %[ pp, fdiff ] = PendentPairNormalFastModified( CovMat, index, init_ind );
        [ pp, fdiff ] = PendentPairNormalFastModified( CovMat, index, init_ind );
        %[ pp1 ] = PendentPair( F, index, 1 );
        %assert( all( pp == pp1 ) )        
    else
        [ pp, fdiff ] = PendentPairNormalFastModified( CovMat, index, init_ind, fdiff );
        %[ pp1 ] = PendentPair( F, index, 1 );
        %assert( all( pp( end ) == pp1( end ) ) )                
    end
    indexrec( i ) = pp( end );
    f( i ) = F( pp{ end } ); 
    index = pp; %% Sort
    index = [ index( 1:end-2 ), { cell2mat( index( end-1:end ) ) } ]; %% Merge
    if i < N - 1 
        fdiff( end-1:end ) = [];
    end
    % fprintf( 'F = %.3f\n', f( i ) );
end
[ tmp, minind ] = min( f );

% further revised on April 26th.
if numel( indexrec{ minind } ) > M/2
    IndexOutput = setdiff( cell2mat( cellfun( @(x) x(:)', index, 'UniformOutput', false ) ), indexrec{ minind } );
else
    IndexOutput = sort( indexrec{ minind } );
end






