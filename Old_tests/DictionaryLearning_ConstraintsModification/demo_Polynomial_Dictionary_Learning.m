% Description: Run file that applies the polynomial dictionary learning algorithm
% in the data contained in testdata.mat. The mat file contains the necessary data that are needed 
% to reproduce the synthetic results of Section V.A.1 of the reference paper:

% D. Thanou, D. I Shuman, and P. Frossard, ?Learning Parametric Dictionaries for Signals on Graphs?, 
% Submitted to IEEE Transactions on Signal Processing,
% Available at:  http://arxiv.org/pdf/1401.0887.pdf

% clear all
close all

load testdata.mat
load reference_dictionary.mat
load initial_sparsity_mx.mat
load SampleSignal.mat

%% Set the parameters

param.N = 100; % number of nodes in the graph
param.S = 4;  % number of subdictionaries 
param.J = param.N * param.S; % total number of atoms 

number_sub = ones(1,param.S);
param.K = 20.*number_sub;
%param.initialDictionary = reference_dictionary;

param.T0 = 4; % sparsity level in the training phase
param.c = 1; % spectral control parameters
param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
param.mu = 1e-2; % polynomial regularizer paremeter

param.percentage = 15; % number of coefficients in the beta vector that I impose to know.
                 % NOTE: is not automatic that #lambdas as root of beta =
                 % percentage, this because while solving the beta
                 % polynomial I can hav invalid roots like complex values
                 % or values not belonging to the interval [0,2]


%% Plot the reandom graph

figure()   
gplot(A,[XCoords YCoords])
 
%% Compute the Laplacian and the normalized Laplacian operator 
    
L = diag(sum(W,2)) - W; % combinatorial Laplacian
param.Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian
[param.eigenMat, param.eigenVal] = eig(param.Laplacian); % eigendecomposition of the normalized Laplacian
[param.lambda_sym,index_sym] = sort(diag(param.eigenVal)); % sort the eigenvalues of the normalized Laplacian in descending order


%% Precompute the powers of the Laplacian

for k=0 : max(param.K)
    param.Laplacian_powers{k + 1} = param.Laplacian^k;
end
    
for j=1:param.N
    for i=0:max(param.K)
        param.lambda_powers{j}(i + 1) = param.lambda_sym(j)^(i);
        param.lambda_power_matrix(j,i + 1) = param.lambda_sym(j)^(i);
     end
end
    
%% Generate the original kernels
output = generate_kernels(param, param.percentage);

%% Generate the original signal
SampleSignal = generate_signal(output, param);

%% Polynomial Dictionary Learning Algorithm 

param.InitializationMethod =  'Random_kernels';
param.displayProgress = 1;
param.numIteration = 8;
param.plot_kernels = 1; % plot the learned polynomial kernels after each iteration
param.quadratic = 0; % solve the quadratic program using interior point methods

% Sparsity matrix initialization
smoothed_TrainSignal = smooth2a(TrainSignal,20,20);
% initial_sparsity_mx = sparsity_matrix_initialize(param,smoothed_TrainSignal);

% Coefficient initialization
param.beta_coefficients = output.coefficients_beta;

%% Dictionary initialization and update step

disp('Starting to train the dictionary');

[Dictionary_Pol, output_Pol, err]  = Polynomial_Dictionary_Learning(SampleSignal, param, initial_sparsity_mx);

%% Dictionary testing step

smoothed_TestSignal = smooth2a(TestSignal,20,20);
CoefMatrix_Pol = OMP_non_normalized_atoms(Dictionary_Pol,smoothed_TestSignal, param.T0);
errorTesting_Pol = sqrt(norm(smoothed_TestSignal - Dictionary_Pol*CoefMatrix_Pol,'fro')^2/size(smoothed_TestSignal,2));
disp(['The total representation error of the testing signals is: ',num2str(errorTesting_Pol)]);

        



