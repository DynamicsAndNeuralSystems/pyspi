function [ pendentPair, f ] = PendentPairNormalFastModified( CovMat, IndexInCell, InitialIndex, FDiffPrevious )
%% [ pendentPair ] = PendentPairNormalFastModified( CovMat, InitialIndex, IndexCell ) 
%% 
%% This function is the core sub-routine of the QueyranneAlgorithmNormal.m
%% 
%% First written by Shohei Hidaka   Mar. 31st, 2016.
%% Revised (added its description in comments) by Shohei Hidaka Jan 13th, 2016.
%% Rebuilt (specialized implementation for the normal distribution) by Shohei Hidaka on April 14th.
%%
%% See also: QueyranneAlgorithm, PendentPair
%%
%% References: Queyranne, M. (1998). Minimizing symmetric submodular functions. Mathematical Programming, 82(1-2), 3-12.
%% http://link.springer.com/article/10.1007%2FBF01585863
%%
%% -- Demo -- 
%% To validate the this code, run the following code:
% CV = cov( randn( 1e3, 10 ) );
% F = @( ind ) TotalCorrDiff( ind, CV );
% index = arrayfun( @(x) x, 1:size( CV, 1 ), 'UniformOutput', false );
% initInd = 1;
% [ PendentPair1 ] = PendentPair( F, index, initInd )
% [ PendentPair2 ] = PendentPairNormal( CV, initInd )
% assert( all( PendentPair1 == PendentPair2 ), 'Two pendent pair is supposed to be identical' )
%

%% Function handles
g = @( Sigma, U1, U2, W ) Sigma( U1, U2 ) - Sigma( U1, W ) / Sigma( W, W ) * Sigma( W, U2 );
PutForwardInCell = @( Cell, key ) [ Cell( key ), Cell( 1:( key - 1 ) ), Cell( ( key + 1 ): end ) ];
logMinor = @( S ) log( diag( eye( size( S, 1 ) ) / S ) ) + sublogdet( S );
InverseUpdate = @( invS, W0, W1 ) invS( W1, W1 ) - invS( W1, W0 )/ invS( W0, W0 ) * invS( W0, W1 );

Sigma = CovMat;
N = size( Sigma, 1 );

S = CovMat ;
invSigma = eye( size( Sigma, 1 ) ) / Sigma ;

%F = @( ind ) TotalCorrDiff( ind, Sigma );
%FDiffMin = zeros( 1, length( IndexInCell ) - 1 );
if nargin < 4 | length( IndexInCell ) == 1
    W = [];
    V = PutForwardInCell( IndexInCell, InitialIndex );
    indMin = 0;
    fmin = zeros( 1, length( V ) - 1 );
else
    indMergedPrevious = IndexInCell( end );
    indToTest = length( IndexInCell );
    indMin = indToTest;

    %%
    fmin = zeros( 1, length( IndexInCell ) - 1 );
    indMin = 0;
    PutForwardInCell = @( Cell, key ) [ Cell( key ), Cell( 1:( key - 1 ) ), Cell( ( key + 1 ): end ) ];
    V = IndexInCell;
    W = [];
    for i = 1:( indToTest - 1 )
        %%
        Vi = IndexInCell{ i };
        W = [ W, Vi ] ; %cell2mat( IndexInCell( 1:i ) ) ;
        V_W = cell2mat( V( ( i + 1 ):end ) );

        S( V_W, V_W ) = g( S, V_W, V_W, Vi );
        invSigma( V_W, V_W ) = InverseUpdate( invSigma, Vi, V_W );  %% Equivalent to: invSigma = inv( Sigma( V_W, V_W ) );

        v = indMergedPrevious{1};
        indc = 1:N; indc( [ v, W ] ) = [];
        fdiff = sublogdet( S( v, v ) ) - sublogdet( S( indc, indc ) )  ...
            - sublogdet( Sigma( v, v ) ) + sublogdet( Sigma( indc, indc ) );
        %% Equivalent to:
            %% fdiff = F( [ IndexInCell( 1:i ), indMergedPrevious ] ) - F( indMergedPrevious )
            %% F = @( ind ) TotalCorrDiff( ind, Sigma );

        if i > length( FDiffPrevious ) | fdiff < FDiffPrevious( i )
            indMin = i;
            break;
        end
    end
    %%%%

    ind = [ 1:( indMin ), indToTest ] ;
    FDiffMin = [ FDiffPrevious( 1:( indMin - 1 ) ), fdiff ];
    if ( indMin + 1 ) == indToTest
        pendentPair = IndexInCell( ind );
        f = FDiffMin;
        return;
    end
    %%
    fmin = FDiffMin;
    V = [ IndexInCell( ind ), IndexInCell( ( indMin + 1 ): end - 1 ) ] ;

end


%% Another implementation of the pendent series

%% http://math.stanford.edu/~lexing/publication/diagonal.pdf
%% "diagonal elements of the inverse matrix"
%% http://math.stackexchange.com/questions/64420/is-there-a-faster-way-to-calculate-a-few-diagonal-elements-of-the-inverse-of-a-h
logSigmaMinor = logMinor( Sigma );
logdiagSigma = log( diag( Sigma ) );
%logdetinvSigmaPrevious = - sublogdet( Sigma );

IsOne = cellfun( @(x) numel(x) == 1, V );

if 1
    i = indMin + 2;
    indWithOne = find( IsOne( i:length( V ) ) ) + i - 1;
    indWithMultiple = find( ~IsOne( i:length( V ) ) ) + i - 1 ;
    v = cell2mat( V( indWithOne ) );
    logSigmaMinor_onV = zeros( 1, length( V ) );
    logSigmaMinor_onV( indWithOne ) = logSigmaMinor( v );
    logdiagSigma_onV = zeros( 1, length( V ) );
    logdiagSigma_onV( indWithOne ) = logdiagSigma( v );
    v = indWithMultiple ;
    for j = v        
        indc = 1:N; indc( V{ j } ) = [];
        logSigmaMinor_onV( j ) = sublogdet( Sigma( indc, indc ) );
        logdiagSigma_onV( j ) = sublogdet( Sigma( V{ j }, V{ j } ) );
    end
end


%V_W_ = cell2mat( V );
for i = ( indMin + 2 ):( length( V ) )
    %for i = 2:length( V )
    Vi = V{ i - 1 };
    W = [ W, Vi ] ;

    indWithOne = find( IsOne( i:length( V ) ) ) + i - 1;
    indWithMultiple = find( ~IsOne( i:length( V ) ) ) + i - 1 ;
    
    % OK?
    V_W = cell2mat( V( i:end ) );
    S( V_W, V_W ) = g( S, V_W, V_W, Vi );
    logdetW = sublogdet( Sigma( W, W ) );

    f = -inf( 1, length( V ) );
    f( i:length( V ) ) = - ( logSigmaMinor_onV( i:length( V ) ) - logdetW ) - logdiagSigma_onV( i:length( V ) );
    %f( indWithMultiple ) = - ( logSigmaMinor_onV( indWithMultiple ) - logdetW ) - logdiagSigma_onV( indWithMultiple );

    if ~isempty( indWithOne )
        %V_W = cell2mat( V( i:end ) );
        %S( V_W, V_W ) = g( S, V_W, V_W, Vi );

        logdiagS = log( diag( S ) );
        %% The following is equivalent to: logVMinor( V_W ) = logMinor( Sigma( V_W, V_W ) );
        invSigma( V_W, V_W ) = InverseUpdate( invSigma, Vi, V_W );  %% Equivalent to: invSigma = inv( Sigma( V_W, V_W ) );

        logVMinor = nan( 1, N );
        %logdetinvSigmaPrevious = logdetinvSigmaPrevious - sublogdet( invSigma( Vi, Vi ) );
        %logVMinor( V_W ) = log( diag( invSigma( V_W, V_W ) ) ) - logdetinvSigmaPrevious;
        %% The above is equivalent to:
        logVMinor( V_W ) = log( diag( invSigma( V_W, V_W ) ) ) + sublogdet( Sigma( V_W, V_W ) );
        %assert( max( abs( logVMinor( V_W ) - logVMinor_' ) ) < 1e-5 );


        %% For cells with one element:
        v = cell2mat( V( indWithOne ) );
        %f( indWithOne ) = logdiagS( v ) - ( logSigmaMinor( v ) - logdetW ) - logdiagSigma( v ) + logVMinor( v )';
        %f( indWithOne ) = f( v ) + logdiagS( v )' + logVMinor( v );
        %assert( max( abs( logSigmaMinor_onV( indWithOne ) - logSigmaMinor( v )' ) ) ) < 1e-5 )
        %assert( max( abs( f( indWithOne ) - ( - ( logSigmaMinor( v ) - logdetW ) - logdiagSigma( v ) )' ) ) < 1e-5 )
        f( indWithOne ) = f( indWithOne ) + logdiagS( v )' + logVMinor( v );
        %( logSigmaMinor( v ) - logdetW ) - ( logSigmaMinor_( v )' - logdetW )
        
        %f1 = @(v) logdiagS( v ) - ( logSigmaMinor( v ) - logdetW ) - logdiagSigma( v ) + logVMinor( v );
        %f( indWithOne ) = cellfun( f1, V( indWithOne ) );
        %assert( max( abs( f( indWithOne ) - gg( indWithOne ) ) ) < 1e-5 );
    end
    
    %% For cells with multiple elements:    
    %f( indWithMultiple ) = - ( logSigmaMinor_onV( indWithMultiple ) - logdetW ) - logdiagSigma_onV( indWithMultiple );
    for j = indWithMultiple %i:length( V )
        %% For indWithOne
        % f( j ) = logdiagS( V{ j } ) - ( logSigmaMinor( V{j} ) - logdetW ) - logdiagSigma( V{ j } ) + logVMinor( V{j} ); %% Fastest
        indc = 1:N; indc( [ V{ j }, W ] ) = [];
        %f( j ) = sublogdet( S( V{ j }, V{ j } ) ) - sublogdet( S( indc, indc ) )  ...
        %    - sublogdet( Sigma( V{ j }, V{ j } ) ) + sublogdet( Sigma( indc, indc ) );
        %f( j ) = sublogdet( S( V{ j }, V{ j } ) ) - ( logSigmaMinor_onV( j ) - logdetW )  ...
        %    - logdiagSigma_onV( j ) + sublogdet( Sigma( indc, indc ) );
        f( j ) = f( j ) + sublogdet( S( V{ j }, V{ j } ) ) + sublogdet( Sigma( indc, indc ) );
    end
    
    [ fmin( i - 1 ), minind ] = min( f( i:length( V ) ) );    
    V( i:end ) = PutForwardInCell( V( i:end ), minind );
    IsOne( i:end ) = PutForwardInCell( IsOne( i:end ), minind );
    logSigmaMinor_onV( i:end ) = PutForwardInCell( logSigmaMinor_onV( i:end ), minind );
    logdiagSigma_onV( i:end ) = PutForwardInCell( logdiagSigma_onV( i:end ), minind );
end
pendentPair = V;
f = fmin;

end

%% 
function [ logdet ] = sublogdet( mat )
% if 0 %size( mat, 1 ) >= 10
%     logdet = sum( log( eig( mat ) ) );
% else
%     logdet = log( det( mat ) );
% end
%% http://blogs.sas.com/content/iml/2012/10/31/compute-the-log-determinant-of-a-matrix.html
% Let A be your matrix and let G = root(A) be the Cholesky root of the matrix A. Then the following equation is true:
%       log(det(A)) = 2*sum(log(vecdiag(G)))
% A = G`*G, by definition of the Cholesky root
% log(det(A)) = log(det(G`*G)) = log(det(G`)*det(G)) = 2*log(det(G))
% Since G is triangular, det(G) = prod(vecdiag(G))
% Therefore log(det(G))=sum(log(vecdiag(G)))
% Consequently, log(det(A)) = 2*sum(log(vecdiag(G)))
%
dim = size( mat, 1 );
if dim == 1
    logdet = log( mat );
elseif dim == 2
    det2 = @( X ) X( 1, 1 ) * X( 2, 2 ) - X( 1, 2 ) * X( 2, 1 );
    logdet = log( det2( mat ) );
elseif dim == 3
    det3 = @( X ) X( 1, 1 ) * X( 2, 2 ) * X( 3, 3 ) + X( 2, 1 ) * X( 3, 2 ) * X( 1, 3 ) + X( 3, 1 ) * X( 1, 2 ) * X( 2, 3 ) ...
        - X( 1, 1 ) * X( 3, 2 ) * X( 2, 3 ) - X( 2, 1 ) * X( 1, 2 ) * X( 3, 3 ) - X( 3, 1 ) * X( 2, 2 ) * X( 1, 3 );
    logdet = log( det3( mat ) );
else
    try
        logdet = 2 * sum( log( diag( chol( mat ) ) ) );
    catch
        logdet = 2 * sum( log( abs( eig( mat ) ) ) );
    end
end

end
