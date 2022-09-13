function [ TC, TCDiffComplete ] = TotalCorrDiff( ind, CV, IC )
% [ TC ] = TotalCorrDiff( X, CV ) 
% For the index set { 1, 2, ..., size( cv, 1 ) }, write its subset by X
% and the complent of X by Y.
% Then, 
% TotalCorrDiff( X, cv ) = TC( X, Y ) - TC( X ) - TC( Y ) + H( X, Y ).
% As H( X, Y ) is fixed when considering the partition (X, Y), it skips the computation.
% Then, for normally distributed variables with the index sets X and Y and its covariance matrix CV, 
% TotalCorrDiff( X, cv ) = log( det( CV( X, X ) ) ) + log( det( CV( Y, Y ) ) )
%
% 
% Input ---
%         X: A subset of index set { 1, .., N }.
%         CV: A covariance matrix for the N variables.
% Output ---
%         TC: The difference in total correlation of { X, Y } to the sum of those of X and Y.
%
% !!!!Note!!!!
% This difference TC misses H( X, Y ), as submodular function optimization does not need this term.
% If one wants to have the proper diff with the constant H( X, Y ),
% receive the second output argument, and have TCDiffComplete = TC - log( det( CV ) ).
% 
% First written by Shohei Hidaka, Oct 22nd 2015
% Revised by Shohei Hidaka, Jan 13th, 2016.
% Revised by Shohei Hidaka, Mar 29th, 2016. (IC option as the third input argument is added)
% 
% See also: MaxTotalCorr
% 

if nargin < 3
    IC = @( DOF ) 0;
end

if iscell( ind )
    ind = cell2mat( ind );
end
indc = setdiff( 1:size( CV, 1 ), ind );

%% Original code -- this may be unstable for a large index set, as
%% its determinant can become vary small and rounded to 0.
% TC = log( det( CV( ind, ind ) ) ) + log( det( CV( indc, indc ) ) ) ; %%

% det1 = det( CV( ind, ind ) );
% det2 = det( CV( indc, indc ) );
% if det1 == 0
%     %det1 = sum( log( abs( eig( CV( ind, ind ) ) ) ) ); %% [DELETE] Bug version since Oct 22nd 2015
%     logdet1 = sum( log( abs( eig( CV( ind, ind ) ) ) ) ); %% [ADD] Revised Jan 13th 2016
% else
%     %% [ADD] Revised Jan 13th 2015
%     logdet1 = log( det1 );
% end
% if det2 == 0
%     %det2 = sum( log( abs( eig( CV( indc, indc ) ) ) ) );  %% [DELETE] Bug version since Oct 22nd 2015
%     logdet2 = sum( log( abs( eig( CV( indc, indc ) ) ) ) ); %% [ADD] Revised Jan 13th 2016
% else
%     %% [ADD] Revised Jan 13th 2015
%     logdet2 = log( det2 );
% end
% 
% TC = logdet1 + logdet2 ;

TC = sub_logdet( CV( ind, ind ) ) + sub_logdet( CV( indc, indc ) );

%% IC (Information criterion) is added
TC = TC + IC( length( ind ) ) + IC( length( indc ) );

if nargout > 1
    %% !!!!Note!!!!
    %% This difference TC misses H( X, Y ), as submodular function optimization does not need this term.
    %% If one wants to have the proper diff with the constant H( X, Y ),
    %% receive the second output argument, and have TC - log( det( CV ) ).
    TCDiffComplete = TC - sub_logdet( CV );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% [ADD] Revised Jan 13th 2016
function [ logdet ] = sub_logdet( CV )
det0 = det( CV );
if det0 < 1e-15
    logdet = sum( log( abs( eig( CV ) ) ) ); 
else
    logdet = log( det0 );
end

