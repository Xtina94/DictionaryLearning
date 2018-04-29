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

% load testdata.mat
% load reference_dictionary.mat
% load SampleSignal.mat
% load initial_dictionary.mat
load SampleWeight.mat
load X_smooth.mat
% % % load SyntheticData.mat

% % % param.Laplacian = Laplacian;
% % % param.eigenMat = eigenVect;
% % % param.eigneVal = eigenVal;
% % % param.lambda_sym = lambda_sym;

SampleSignal= X_smooth(:,1:900);
TestSignal = X_smooth(:,901:1000);

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
param.epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
param.mu = 1e-2; % polynomial regularizer paremeter


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

%------------------------------------------------------------ 
%%------ Precompute the powers of the Laplacian -------------
%------------------------------------------------------------ 
for k=0 : max(param.K)
    param.Laplacian_powers{k + 1} = param.Laplacian^k;
end
    
for j=1:param.N
    for i=0:max(param.K)
        param.lambda_powers{j}(i + 1) = param.lambda_sym(j)^(i);
        param.lambda_power_matrix(j,i + 1) = param.lambda_sym(j)^(i);
     end
end
    
         
%%---------------------------------------------------------
param.InitializationMethod =  'Random_kernels';
%%---- Polynomial Dictionary Learning Algorithm -----------
%%---------------------------------------------------------
% param.initial_dictionary = initial_dictionary;
param.displayProgress = 1;
param.numIteration = 8;
param.plot_kernels = 1; % plot the learned polynomial kernels after each iteration
param.quadratic = 0; % solve the quadratic program using interior point methods

disp('Starting to train the dictionary');

[Dictionary_Pol, output_Pol]  = Polynomial_Dictionary_Learning(SampleSignal, param);

%%%%%%%%%%%%%%%Parte riguardante il segnale campione%%%%%%%%%%%%%
% % % 
% % % output_CoefKernels = generate_kernels(param);
% % % param.TrainSignal = TrainSignal;
% % %  [SampleSignal, initial_dictionary] = generate_signal(output_CoefKernels,param);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CoefMatrix_Pol = OMP_non_normalized_atoms(Dictionary_Pol,TestSignal, param.T0);
errorTesting_Pol = sqrt(norm(TestSignal - Dictionary_Pol*CoefMatrix_Pol,'fro')^2/size(TestSignal,2));
disp(['The total representation error of the testing signals is: ',num2str(errorTesting_Pol)]);

        



