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
load initial_sparsity_mx_uber
load TestSignal.mat
load learned_W.mat
load uber.mat
load lat.mat
load lon.mat
param.percentage = 12;

%% Compute the Laplacian and the normalized Laplacian operator 

L = param.Laplacian;
W = learned_W;
param.Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian
[param.eigenMat, param.eigenVal] = eig(param.Laplacian); % eigendecomposition of the normalized Laplacian
[param.lambda_sym,index_sym] = sort(diag(param.eigenVal)); % sort the eigenvalues of the normalized Laplacian in descending order

%%------ Precompute the powers of the Laplacian -------------

for k=0 : max(param.K)
    param.Laplacian_powers{k + 1} = param.Laplacian^k;
end
    
for j=1:param.N
    for i=0:max(param.K)
        param.lambda_powers{j}(i + 1) = param.lambda_sym(j)^(i);
        param.lambda_power_matrix(j,i + 1) = param.lambda_sym(j)^(i);
     end
end
    
param.InitializationMethod =  'Random_kernels';
SampleSignal = param.y;
param.initial_sparsity_mx_uber = initial_sparsity_mx_uber;
% % % param.initial_sparsity_mx = sparsity_matrix_initialize(param,SampleSignal);

%%---- Polynomial Dictionary Learning Algorithm -----------


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

% % % smoothed_signal = smooth2a(TestSignal,15,15);

CoefMatrix_Pol = OMP_non_normalized_atoms(Dictionary_Pol,TestSignal(1:29,:), param.T0);
errorTesting_Pol = sqrt(norm(TestSignal(1:29,:) - Dictionary_Pol*CoefMatrix_Pol,'fro')^2/size(TestSignal,2));
disp(['The total representation error of the testing signals is: ',num2str(errorTesting_Pol)]);

        


