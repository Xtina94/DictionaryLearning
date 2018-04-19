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

load testdata.mat
load initial_sparsity_mx.mat
load SampleSignal.mat
load initial_dictionary.mat

%------------------------------------------------------    
%%---- Set the paremeters-------- 
%------------------------------------------------------

param.N = 100; % number of nodes in the graph
param.S = 4;  % number of subdictionaries 
param.J = param.N * param.S; % total number of atoms 

%%% My changings %%%
number_sub = ones(1,param.S);
param.K = 20.*number_sub;
% % % param.initialDictionary = reference_dictionary;
%%%

%param.K = [20 20 20 20]; % polynomial degree of each subdictionary
param.T0 = 4; % sparsity level in the training phase
param.c = 1; % spectral control parameters
param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
param.mu = 1e-2; % polynomial regularizer paremeter
param.initial_dictionary = initial_dictionary;

param.percentage = 15;

%------------------------------------------------------    
%%---- Plot the random graph-------- 
%------------------------------------------------------
% % % figure()   
% % % gplot(A,[XCoords YCoords])

%------------------------------------------------------------  
%%- Compute the Laplacian and the normalized Laplacian operator 
%------------------------------------------------------------
    
L = diag(sum(W,2)) - W; % combinatorial Laplacian
param.Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian
[param.eigenMat, param.eigenVal] = eig(param.Laplacian); % eigendecomposition of the normalized Laplacian
[param.lambda_sym,index_sym] = sort(diag(param.eigenVal)); % sort the eigenvalues of the normalized Laplacian in descending order

% % % smoothed_signal = smooth_signal(TestSignal, L);

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
    
param.InitializationMethod =  'GivenMatrix';

%%---- Polynomial Dictionary Learning Algorithm -----------

% % % param.initial_dictionary = initial_dictionary;
param.displayProgress = 1;
param.numIteration = 8;
param.plot_kernels = 1; % plot the learned polynomial kernels after each iteration
param.quadratic = 0; % solve the quadratic program using interior point methods

disp('Starting to train the dictionary');

[Dictionary_Pol, output_Pol]  = Polynomial_Dictionary_Learning(SampleSignal, param);

CoefMatrix_Pol = OMP_non_normalized_atoms(Dictionary_Pol,TestSignal, param.T0);
errorTesting_Pol = sqrt(norm(TestSignal - Dictionary_Pol*CoefMatrix_Pol,'fro')^2/size(TestSignal,2));
disp(['The total representation error of the testing signals is: ',num2str(errorTesting_Pol)]);


%% Generating the SampleSignal

% % % load SampleSignal.mat
% % % SampleSignal = Dictionary_Pol*output_Pol.CoefMatrix;
        



