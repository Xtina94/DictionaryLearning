 %==========================================================================
     %% Example
%==========================================================================

% Description: Run file that applies the polynomial dictionary learning algorithm
% in the data contained in testdata.mat. The mat file contains the necessary data that are needed 
% to reproduce the synthetic results of Section V.A.1 of the reference paper:

% D. Thanou, D. I Shuman, and P. Frossard, ?Learning Parametric Dictionaries for Signals on Graphs?, 
% Submitted to IEEE Transactions on Signal Processing,
% Available at:  http://arxiv.org/pdf/1401.0887.pdf

close all
load learned_dictionary_uber.mat
load UberData.mat

param.N = size(X,1); % number of nodes in the graph
param.S = 2;  % number of subdictionaries 
param.J = param.N * param.S; % total number of atoms 
number_sub = ones(1,param.S);
param.K = 15.*number_sub;
param.T0 = 4; % sparsity level in the training phase
param.c = 1; % spectral control parameters
param.epsilon = 0.05; % we assume that epsilon_1 = epsilon_2 = epsilon
param.mu = 1e-2; % polynomial regularizer paremeter

param.percentage = 9;
param.signal = X(:,1:80);
TrainSignal = param.signal;
TestSignal = X(:,81:size(X,2));

%% Compute the Laplacian and the normalized Laplacian operator 

W = learned_W;
L = diag(sum(W,2)) - W; % combinatorial Laplacian
param.Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian
[param.eigenMat, param.eigenVal] = eig(param.Laplacian); % eigendecomposition of the normalized Laplacian
[param.lambda_sym,index_sym] = sort(diag(param.eigenVal)); % sort the eigenvalues of the normalized Laplacian in descending order

%% Precompute the powers of the Laplacian

for k = 0 : max(param.K)
    param.Laplacian_powers{k + 1} = param.Laplacian^k;
end
    
for j=1:param.N
    for i=0:max(param.K)
        param.lambda_powers{j}(i + 1) = param.lambda_sym(j)^(i);
        param.lambda_power_matrix(j,i + 1) = param.lambda_sym(j)^(i);
     end
end
    
param.InitializationMethod =  'Random_kernels';
param.initial_dictionary_uber = learned_dictionary;

%% Polynomial Dictionary Learning Algorithm

param.displayProgress = 1;
param.numIteration = 8;
param.plot_kernels = 1; % plot the learned polynomial kernels after each iteration
param.quadratic = 0; % solve the quadratic program using interior point methods

disp('Starting to train the dictionary');

[Dictionary_Pol, output_Pol]  = Polynomial_Dictionary_Learning(TrainSignal, param);

CoefMatrix_Pol = OMP_non_normalized_atoms(Dictionary_Pol,TestSignal(1:29,:), param.T0);
errorTesting_Pol = sqrt(norm(TestSignal(1:29,:) - Dictionary_Pol*CoefMatrix_Pol,'fro')^2/size(TestSignal,2));
disp(['The total representation error of the testing signals is: ',num2str(errorTesting_Pol)]);
sum_kernels = sum(output_Pol.g_ker,2);

%% Save results to file
filename = 'Output_results_Uber.mat';
totalError = output_Pol.totalError;
alpha_coeff = output_Pol.alpha;
g_ker = output_Pol.g_ker;
save(filename,'Dictionary_Pol','totalError','alpha_coeff', 'g_ker','CoefMatrix_Pol','errorTesting_Pol','TrainSignal','TestSignal','sum_kernels');

% filename2 = strcat('m',num2str(param.percentage),'_e',num2str(param.epsilon*100),'_thr53.pdf');
% fileID = fopen(filename2,'w');
% fprintf(fileID,num2str(errorTesting_Pol));
% fclose(fileID);




