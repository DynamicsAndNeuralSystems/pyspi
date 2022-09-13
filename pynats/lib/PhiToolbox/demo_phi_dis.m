clear

addpath(genpath('../PhiToolbox'))

%% parameters for computing phi
Z = [1 2]; % partition
tau = 1; % time delay
N_s = 2;  % number of states

%% important
params.tau = tau;
params.number_of_states = N_s;

%% generate time series from Boltzman machine
N = 2; % number of units
T = 10^6; % number of iterations
W = [0.1 0.6; 0.4 0.2]; % connectivity matrix
beta = 4; % inverse temperature
X = generate_Boltzmann(beta,W,N,T); 

%% compute phi from time series data X
options.type_of_dist = 'discrete';
% type_of_dist:
%    'Gauss': Gaussian distribution
%    'discrete': discrete probability distribution

% available options of type_of_phi for discrete distributions:
%    'MI1': Multi (Mutual) information, e.g., I(X_1; X_2). (IIT1.0)
%    'SI': phi_H, stochastic interaction
%    'star': phi_star, based on mismatched decoding


% options.type_of_phi = 'MI1';
options.type_of_phi = 'star';
phi = phi_comp(X, Z, params, options);

%% show the resullts
fprintf('partition\n')
disp(Z);
fprintf('phi=%f\n',phi);