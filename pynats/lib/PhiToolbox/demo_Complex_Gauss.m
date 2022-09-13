%% Find the complex in a multivariate autoregressive model, 
%   X^t = A*X^{t-1} + E,
% where A is a connectivity matrix and E is gaussian noise.

addpath(genpath('../PhiToolbox'))

%% generate data
disp('Generating data...')

% construct connectivity matrix
N = 6; % N: the number of elements
A = zeros(N); % A: the connextivity Matrix

A(1:2, 1:2) = 0.05;
A(3:6, 3:6) = 0.05;
A(5:6, 5:6) = 0.1;
A(1:(N+1):end) = 0.9; % self connections

A(1,3) = 0.01;
A(3,1) = 0.01;
A(2,4) = 0.01;
A(4,2) = 0.01;

A = A/N;

figure, imagesc(A), axis equal tight
title('Connectivity Matrix', 'FontSize', 18)
colorbar

% generate time series X using the AR model
T = 10^7;
X = zeros(N,T);
X(:,1) = randn(N,1);
for t=2: T
    E = randn(N,1);
    X(:,t) = A*X(:,t-1) + E;
end

%% params
params.tau = 1; % time lag

%% options
%%% Example 1
options.type_of_dist = 'Gauss'; % type of probability distributions
options.type_of_phi = 'MI1'; % type of phi
options.type_of_MIPsearch = 'Queyranne'; % type of MIP search
options.type_of_complexsearch = 'Recursive'; % type of complex search
options.normalization = 0; % normalization of phi by Entropy

%%%% Example2
% options.type_of_dist = 'Gauss'; % type of probability distributions
% options.type_of_phi = 'star'; % type of phi
% options.type_of_MIPsearch = 'Queyranne'; % type of MIP search
% options.type_of_complexsearch = 'Exhaustive'; % type of complex search
% options.normalization = 0; % normalization of phi by Entropy

%% Find complexes and main complexes
[complexes, phis_complexes, main_complexes, phis_main_complexes, Res] = ...
    Complex_search( X, params, options );

%% Show results
main_complexes_str = cell(size(main_complexes));
for i = 1:length(main_complexes)
    main_complexes_str{i} =  num2str(main_complexes{i});
end
figure
bar(phis_main_complexes)
set(gca, 'xticklabel', main_complexes_str)
title('Main Complexes')
ylabel('\Phi^{MIP}')
xlabel('Indices of the main complexes')


XCoor = [0 0 1 1 2 2];
YCoor = [1 0 1 0 1 0];
figure
colormap(KovesiRainbow)
plotComplexes(complexes,phis_complexes,'BirdsEye','XData',XCoor,'YData',YCoor, 'LineWidth', 2)
title('(Main) Complexes')
colorbar

min_phi_comp = min(phis_complexes);
max_phi_comp = max(phis_complexes);
zlim([min_phi_comp, max_phi_comp])
zlabel('$\Phi^\mathrm{MIP}$', 'Interpreter', 'latex', 'FontSize', 18)

view(-6.8, 11.6)
