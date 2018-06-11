%%
clear all
close all
addpath(genpath(pwd))

% load data in variable X here (it should be a matrix #nodes x #signals)

%% For synthetic data
% % % load testdata.mat
% % % param.S = 4;  % number of subdictionaries 
% % % param.K = [20 20 20 20]; % polynomial degree of each subdictionary
% % % param.percentage = 10;

%% For Uber
load UberData.mat
TestSignal = X(:,91:110);
TrainSignal = X(:,1:90);
param.S = 2;  % number of subdictionaries 
param.K = [15 15]; % polynomial degree of each subdictionary
param.percentage = 8;

X = TrainSignal;
param.N = size(X,1); % number of nodes in the graph
param.J = param.N * param.S; % total number of atoms 
K = max(param.K);
param.c = 1; % spectral control parameters
param.epsilon = 0.55; %0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
param.mu = 1;%1e-2; % polynomial regularizer paremeter
param.y = X; %signals
param.y_size = size(param.y,2);
param.T0 = 6; %sparsity level (# of atoms in each signals representation)

%% initialise learned data

% Fixed stuff
[param.Laplacian, learned_W] = init_by_weight(param.N);
alpha_gradient = 2; %gradient descent parameter, it decreases with epochs

% 1.Case: Start from graph learning - (generate dictionary polynomial coefficients from heat kernel)
param.t = param.K;
for i = 1 : param.S
    param.t(i) = param.S-(i-1); %heat kernel coefficients; this heat kernel will be inverted to cover high frequency components
end
param.alpha = generate_coefficients(param);
disp(param.alpha);
[learned_dictionary, param] = construct_dict(param);

% 2: Case: Start from Dictionary learning - (Initialize the dictionary and the alpha coefficients' structure)
% % % [param.eigenMat, param.eigenVal] = eig(param.Laplacian); % eigendecomposition of the normalized Laplacian
% % % [learned_dictionary] = initialize_dictionary(param);
% % % [param.lambda_sym,index_sym] = sort(diag(param.eigenVal)); % sort the eigenvalues of the normalized Laplacian in descending order
% % % [param.beta_coefficients, param.rts] = retrieve_betas(param);


for big_epoch = 1:10
    %% optimise with regard to x
    disp(['Epoch... ',num2str(big_epoch)]);
    x = OMP_non_normalized_atoms(learned_dictionary,param.y, param.T0);
    % For Case2:
    if mod(big_epoch,2) == 0
    % For case 1:
%     if mod(big_epoch,2) ~= 0
        %optimise with regard to W
        maxEpoch = 1; %number of graph updating steps before updating sparse codes (x) again
        beta = 10^(-2); %graph sparsity penalty
        old_L = param.Laplacian;
        [param.Laplacian, learned_W] = update_graph(x, alpha_gradient, beta, maxEpoch, param,learned_W, learned_W);
        [learned_dictionary, param] = construct_dict(param);
        alpha_gradient = alpha_gradient*0.985; %gradient descent decreasing
    else
        %Optimize with regard to alpha
        K = max(param.K);
        [param.eigenMat, param.eigenVal] = eig(param.Laplacian); % eigendecomposition of the normalized Laplacian
        [param.lambda_sym,index_sym] = sort(diag(param.eigenVal)); % sort the eigenvalues of the normalized Laplacian in descending order
        [param.beta_coefficients, param.rts] = retrieve_betas(param);
        param.lambda_power_matrix = zeros(param.N,K+1);
        for i = 0:K
            param.lambda_power_matrix(:,i+1) = param.lambda_sym.^i;
        end
        for k = 0 : max(param.K)
            param.Laplacian_powers{k + 1} = param.Laplacian^k;
        end

        param.alpha_vector = polynomial_construct_low(param);         
        temp_alpha = coefficient_update_interior_point(param.y,x,param,'sdpt3');
        for j = 1:param.S
            param.alpha{j} = temp_alpha((j-1)*(K+1)+1:j*(K+1))';
        end
        [learned_dictionary, param] = construct_dict(param);
    end
    
end

%% Plot the kernels

g_ker = zeros(param.N, param.S);
for i = 1 : param.S
    for n = 1 : param.N
        p = 0;
        for l = 0 : param.K(i)
            p = p +  param.alpha{i}(l+1)*param.lambda_power_matrix(n,l + 1);
        end
        g_ker(n,i) = p;
    end
end

param.kernel = g_ker;

figure()
hold on
for s = 1 : param.S
    plot(param.lambda_sym(2:length(param.lambda_sym)),g_ker(2:length(param.lambda_sym),s));
end
hold off

filename = strcat('Uber_Kernels','_e',num2str(param.epsilon*100),'_m',num2str(param.percentage));
saveas(gcf,filename,'png');

CoefMatrix_Pol = OMP_non_normalized_atoms(learned_dictionary,TestSignal, param.T0);
errorTesting_Pol = sqrt(norm(TestSignal - learned_dictionary*CoefMatrix_Pol,'fro')^2/size(TestSignal,2));
disp(['The total representation error of the testing signals is: ',num2str(errorTesting_Pol)]);
sum_kernels = sum(param.kernel,2);

%%
%constructed graph needs to be tresholded, otherwise it's too dense
%fix the number of desired edges here at nedges
nedges = 4*29;
final_Laplacian = treshold_by_edge_number(param.Laplacian, nedges);
final_W = learned_W.*(final_Laplacian~=0);

%% Save results to file
filename = 'Output_results_Uber';
alpha_coeff = zeros(K+1,2);
for i = 1:param.S
    alpha_coeff(:,i) = param.alpha{i};
end
save(filename,'final_Laplacian','final_W','alpha_coeff', 'g_ker','CoefMatrix_Pol','errorTesting_Pol','TrainSignal','TestSignal','sum_kernels','learned_dictionary');

