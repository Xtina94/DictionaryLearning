%%
clear all
close all

%% Adding the paths
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\Optimizers'); %Folder conatining the yalmip tools
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\Comparison_datasets\'); %Folder containing the copmarison datasets
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\'); %Folder containing the training and verification dataset
path = 'C:\Users\Cristina\Documents\GitHub\OrganizedFiles\Graph&DictLearning\GraphLearning\Results\'; %Folder containing the results to save

flag = 1;
switch flag
    case 1 %Dorina
        load DataSetDorina.mat
        load ComparisonDorina.mat
    case 2 %Uber
        load DataSetUber.mat
        load ComparisonUber.mat
    case 3 %Cristina
        load DatasetLF.mat
        load ComparisonLF.mat;
end

switch flag
    case 1 %Dorina
        Y = TrainSignal;
        K = 20;
        param.S = 4;  % number of subdictionaries         
        param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 20;
        param.N = 100; % number of nodes in the graph
        ds = 'Dataset used: Synthetic data from Dorina';
        ds_name = 'Dorina';
        param.percentage = 15;
        param.thresh = param.percentage+60;
    case 2 %Cristina
        Y = TrainSignal;
        K = 15;
        param.S = 2;  % number of subdictionaries        
        param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
        param.N = 100; % number of nodes in the graph
        ds = 'Dataset used: data from Cristina';
        ds_name = 'Cristina'; 
        param.percentage = 8;
        param.thresh = param.percentage+60;
    case 3 %Uber
        Y = TrainSignal;
        K = 15;
        param.S = 2;  % number of subdictionaries 
        param.epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
        param.N = 29; % number of nodes in the graph
        ds = 'Dataset used: data from Uber';
        ds_name = 'Uber';
        param.percentage = 8;
        param.thresh = param.percentage+6;
end

param.N = size(Y,1); % number of nodes in the graph
param.J = param.N * param.S; % total number of atoms 
param.K = K*ones(1,param.S); % polynomial degree of each subdictionary
param.c = 1; % spectral control parameters
param.epsilon = 0.05;%0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
param.mu = 1;%1e-2; % polynomial regularizer paremeter
param.y = Y; %signals
param.y_size = size(param.y,2);
param.T0 = 4; %sparsity level (# of atoms in each signals representation)

%% generate dictionary polynomial coefficients from heat kernel

for i = 1:param.S
    if mod(i,2) ~= 0
        param.t(i) = 2; %heat kernel coefficients
    else
        param.t(i) = 1; % Inverse of the heat kernel coefficients
    end
end
param.alpha = generate_coefficients(param);
disp(param.alpha);

for i = 1:param.S
    param.alpha{i} = comp_alpha((i-1)*(K+1) + 1:i*(K+1),1);
end

%% initialise learned data: in our case we start from learning alphas so we initialize the dictionary
[param.Laplacian, initial_W] = init_by_weight(param.N);
[initial_dictionary, param] = construct_dict(param); %Saved laplacian powers and lambda powers here
grad_desc = 2; %gradient descent parameter, it decreases with epochs

for big_epoch = 1:5      
    if big_epoch == 1
        learned_dictionary = initial_dictionary;
        learned_W = initial_W;
    end
    
    %% optimise with regard to x
    disp(['Epoch... ',num2str(big_epoch)]);
    X = OMP_non_normalized_atoms(learned_dictionary,param.y, param.T0);
    
    %% optimize with regard to alphas
    if mod(big_epoch,2) ~= 0
    %% optimise with regard to W
        maxEpoch = 1; %number of graph updating steps before updating sparse codes (x) again
        beta = 10^(-2); %graph sparsity penalty
        old_L = param.Laplacian;
        [param.Laplacian, learned_W] = update_graph(X, grad_desc, beta, maxEpoch, param, learned_dictionary, learned_W);
        [learned_dictionary, param] = construct_dict(param);
        grad_desc = grad_desc*0.985; %gradient descent decreasing
    end
end

%% At the end of the cycle I have:
% param.alpha --> the learned coefficients;
% X           --> the learned sparsity mx;
% learned_W   --> the learned W from the old D and alpha coeff;
% learned_dictionary --> the learned final dictionary;
% cpuTime     --> the final cpuTime

%% Estimate the final reproduction error

X = OMP_non_normalized_atoms(learned_dictionary,TestSignal, param.T0);
errorTesting_Pol = sqrt(norm(TestSignal - learned_dictionary*X,'fro')^2/size(TestSignal,2));
disp(['The total representation error of the testing signals is: ',num2str(errorTesting_Pol)]);

%%
%constructed graph needs to be tresholded, otherwise it's too dense
%fix the number of desired edges here at nedges
nedges = 4*29;
final_Laplacian = treshold_by_edge_number(param.Laplacian, nedges);
final_W = learned_W.*(final_Laplacian~=0);

%% Last eigenDecomposition, needed to compare the norm of the lambdas

[param.eigenMat, param.eigenVal] = eig(final_Laplacian);
[param.lambda_sym,index_sym] = sort(diag(param.eigenVal));

%% Compute the l-2 norms

lambda_norm = norm(comp_eigenVal - param.lambda_sym);
alpha_norm = '0 since the alphas do not change here'; %norm(comp_alpha - param.alpha);
X_norm = norm(comp_X - X);
D_norm = norm(comp_D - learned_dictionary);
W_norm = norm(comp_W - learned_W);
W_norm_thr = norm(comp_W - final_W); %The thresholded adjacency matrix

%% Save the results to file

% The norms
filename = [path,'Norms.mat'];
save(filename,'lambda_norm','alpha_norm','X_norm','D_norm','W_norm');

% The Output data
filename = [path,'Output.mat'];
learned_eigenVal = param.lambda_sym;
save(filename,'ds','learned_dictionary','learned_W','final_W','X','learned_eigenVal','errorTesting_Pol');

%% Verifying the results with the precision recall function
% % %     learned_Laplacian = param.Laplacian;
% % %     [optPrec, optRec, opt_Lapl] = precisionRecall(true_Laplacian, learned_Laplacian);
% % %     filename = [path,'ouput_PrecisionRecall_attempt',num2str(attempt_index),'.mat'];
% % %     save(filename,'opt_Lapl','optPrec','optRec');
