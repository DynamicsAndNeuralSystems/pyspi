function map = KovesiRainbow(n)

if nargin < 1
    n = 64;
end
map = colorcet('R1', 'N', n);

end