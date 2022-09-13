function sigma = base10toM(C,N,M)

%-------------------------------------------------------------------------------------------------
% PURPOSE: convert base 10 to base M
%
% INPUTS:
%   C: decimal value
%   N: the number of digits
%   M: base
%
% OUTPUTS:
%  sigma: base M value
%-------------------------------------------------------------------------------------------------
%
% Masafumi Oizumi, 2018

% assert( isa( C, 'double' ) );
% assert( isa( N, 'double' ) );
% assert( isa( M, 'double' ) );

% assert( all( size(C) == [1 1] ) );
% assert( all( size(N) == [1 1]) ); 
% assert( all( size(M) == [1 1]) ); 

sigma = zeros(N,1);

for i=1: N
    sigma(i) = C-floor(C/M)*M;
    C = floor(C/M);
end

end