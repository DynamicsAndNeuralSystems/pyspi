clear all;

addpath(genpath('../PhiToolbox'))

N = 4; % number of units
T = 10^6; % number of iterations
tau = 1; % time delay
N_st = 2;  % number of states

%% params
params.tau = tau;
params.number_of_states = N_st;

%% generate time series from Boltzman machine
beta = 4; % inverse temperature

% connectivity matrix
W = zeros(N,N);
community = [1 2 2 1];
for i=1: N
    for j=1: N
        if i~=j
            if community(i) == community(j)
                % W(i,j) = 0.2; % for N = 8
                W(i,j) = 0.4;
            else
                W(i,j) = 0;
            end
        else
            W(i,i) = 0.1;
        end
    end
end
X = generate_Boltzmann(beta,W,N,T); 

%% 

T_seg = 1000;
figure(1)
t_vec1 = 1: T_seg;
t_vec2 = 2*10^3: 2*10^3+T_seg;
t_vec3 = 10^4: 10^4+T_seg;
t_vec4 = 10^5: 10^5+T_seg;
t_vec5 = T-300: T;
subplot(3,2,1),imagesc(X(:,t_vec1));
subplot(3,2,2),imagesc(X(:,t_vec2));
subplot(3,2,3),imagesc(X(:,t_vec3));
subplot(3,2,4),imagesc(X(:,t_vec4));
subplot(3,2,5),imagesc(X(:,t_vec5));

%% compute correlation 
R = corrcoef(X');
disp('Correlation Matrix')
disp(R);

%% options
options.type_of_dist = 'discrete';
options.type_of_phi = 'star';
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
