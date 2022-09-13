%% Find Minimum Information Partition (MIP) in a multivariate autoregressive model, 
%   X^t = A*X^{t-1} + E,
% where A is a connectivity matrix and E is gaussian noise.

clear
addpath(genpath('../PhiToolbox'))

%% generate time series data
disp('Generating time series data...')

%%% construct connectivity matrix %%%
N = 10; % the number of elements
A = zeros(N); % A: connectivity matrix (block structured)
n_block = 2;
for i = 1:n_block
    indices_block = ((1+(i-1)*N/n_block):i*N/n_block)';
    A(indices_block, indices_block) = 1/N;
end
A = A + 0.01*randn(N)/sqrt(N);

figure(1)
imagesc(A)
title('Connectivity Matrix A')
colorbar;

%%% generate random gaussian time series X %%%
T = 10^6;
X = zeros(N,T);
X(:,1) = randn(N,1);
for t=2: T
    E = randn(N,1);
    X(:,t) = A*X(:,t-1) + E;
end

%% params
params.tau = 1; % time delay

%% options
options.type_of_dist = 'Gauss'; % type of probability distributions
options.type_of_phi = 'star'; % type of phi
options.type_of_MIPsearch = 'Queyranne'; % type of MIP search
options.normalization = 0; % normalization by Entropy

%% find Minimum Information Partition (MIP)
tic;
[Z_MIP, phi_MIP] = MIP_search(X, params, options );
CalcTime = toc;

disp( ['CalcTime=', num2str(CalcTime)])
disp(['phi at the MIP: ', num2str(phi_MIP)])
disp(['the MIP: ', num2str(Z_MIP)])
disp(' ')