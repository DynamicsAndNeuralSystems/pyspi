function x = baseMto10_table(sigma, pow_vec)

%-------------------------------------------------------------------------------------------------
% PURPOSE: convert base M to base 10
%
% INPUTS:
%   sigma: base M value
%   M: base
%
% OUTPUTS:
%  x: decimal value
%-------------------------------------------------------------------------------------------------
%
% Masafumi Oizumi, 2018

% assert( isa( sigma, 'double' ) );
% assert( isa( M, 'double' ) );

% N_max = 100;
% assert( all( size(sigma) <= [N_max 1] ) );
% assert( all( size(M) == [1 1]) ); 

N = length(sigma);

x = 0;
for i=1: N
    x = x + pow_vec(i)*sigma(i);
end