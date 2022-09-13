%% Find the complex in a Boltzmann machine, 
%   input = beta*W*x;
%   prob(x=1) = sigmoid(input);
% where W is a connectivity matrix

clear
addpath(genpath('../PhiToolbox'))

%% generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Generating data...')

Z = [1 2 3 3 2 1];
N = length(Z); % number of units
T = 10^6; % number of iterations

W = zeros(N,N);

for i=1: N
    for j=1: N
        if i~=j
            if Z(i) == Z(j)
                % W(i,j) = 0.2; % for N=8
                W(i,j) = 0.4; % for N=4
            else
                W(i,j) = 0;
            end
        else
            W(i,i) = 0.1;
        end
    end
end

beta = 4; % inverse temperature

X = generate_Boltzmann(beta,W,N,T); % generate time series of Boltzman machine

%% 

T_seg = 1000;
figure
t_vec1 = 1: T_seg;
t_vec2 = 2*10^3: 2*10^3+T_seg;
t_vec3 = 10^4: 10^4+T_seg;
t_vec4 = 10^5: 10^5+T_seg;
t_vec5 = T-300: T;
subplot(3,2,1),imagesc(X(:,t_vec1));title('Time Series 1')
subplot(3,2,2),imagesc(X(:,t_vec2));title('Time Series 2')
subplot(3,2,3),imagesc(X(:,t_vec3));title('Time Series 3')
subplot(3,2,4),imagesc(X(:,t_vec4));title('Time Series 4')
subplot(3,2,5),imagesc(X(:,t_vec5));title('Time Series 5')

%% compute correlation
R = corrcoef(X');
disp('Correlation Matrix')
disp(R);

figure
[Z_sort, s_ind] = sort(Z);
imagesc(R(s_ind,s_ind));
set(gca, 'XTick', [1: 1: N], 'XTickLabel', s_ind); 
set(gca, 'YTick', [1: 1: N], 'YTickLabel', s_ind); 
title('Correlation Matrix')

%% find the complex
options.type_of_dist = 'discrete';
options.type_of_phi = 'MI1';
options.type_of_MIPsearch = 'Queyranne';
options.type_of_complexsearch = 'Exhaustive';
options.normalization = 0;% normalization of phi by Entropy

params.tau = 1; % time delay
params.number_of_states = 2;  % number of states

%% Find complexes and main complexes
[complexes, phis_complexes, main_complexes, phis_main_complexes, Res] = ...
   Complex_search( X, params, options);

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


figure
colormap(KovesiRainbow)
plotComplexes(complexes,phis_complexes,'BirdsEye','N',N, 'LineWidth', 2)
title('(Main) Complexes')
colorbar

min_phi_comp = min(phis_complexes);
max_phi_comp = max(phis_complexes);
zlim([min_phi_comp, max_phi_comp])
zlabel('$\Phi^\mathrm{MIP}$', 'Interpreter', 'latex', 'FontSize', 18)

view(-6.8, 11.6)



