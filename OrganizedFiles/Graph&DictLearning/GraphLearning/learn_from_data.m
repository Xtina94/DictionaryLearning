%%
clear all
close all

%% Adding the paths
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\Optimizers'); %Folder conatining the yalmip tools
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\Comparison_datasets\'); %Folder containing the copmarison datasets
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\'); %Folder containing the training and verification dataset
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\GeneratingKernels\Results'); %Folder conatining the heat kernel coefficietns
path = 'C:\Users\Cristina\Documents\GitHub\OrganizedFiles\Graph&DictLearning\GraphLearning\Results\'; %Folder containing the results to save

%%
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
    case 4 %1 Heat kernel
        load DataSetHeat.mat;
        load ComparisonHeat.mat;
        load LF_heatKernel.mat;
end

switch flag
    case 1 %Dorina
        X = TrainSignal;
        K = 20;
        param.S = 4;  % number of subdictionaries         
        param.epsilon = 0.05; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 20;
        ds = 'Dataset used: Synthetic data from Dorina';
        ds_name = 'Dorina';
    case 2 %Uber
        X = TrainSignal;
        K = 15;
        param.S = 2;  % number of subdictionaries 
        param.epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
        ds = 'Dataset used: data from Uber';
        ds_name = 'Uber';
    case 3 %Cristina
        X = TrainSignal;
        K = 15;
        param.S = 2;  % number of subdictionaries        
        param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
        ds = 'Dataset used: data from Cristina';
        ds_name = 'Cristina';       
    case 4 %Heat kernel
        X = TrainSignal;
        K = 15;
        param.S = 1;  % number of subdictionaries 
        param.epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
        ds = 'Dataset used: data from Heat kernel';
        ds_name = 'Heat';
end

param.N = size(X,1); % number of nodes in the graph
param.J = param.N * param.S; % total number of atoms 
param.K = K*ones(1,param.S); %[20 20 20 20]; % polynomial degree of each subdictionary
param.c = 1; % spectral control parameters
param.mu = 1;%1e-2; % polynomial regularizer paremeter
param.y = X; %signals
param.y_size = size(param.y,2);
param.T0 = 6; %sparsity level (# of atoms in each signals representation)

%% generate dictionary polynomial coefficients from heat kernel

if flag == 1
    for i = 1:param.S
    param.alpha{i} = comp_alpha((i-1)*(max(param.K)+1) + 1:(max(param.K)+1)*i);
    end
else
    param.t(1) = 2; %heat kernel coefficients
    param.t(2) = 1; %this heat kernel will be inverted to cover high frequency components
    param.alpha = generate_coefficients(param);
    disp(param.alpha);
end

%% Initialize W:

[param.Laplacian, initial_W] = init_by_weight(param.N);
initial_Laplacian = param.Laplacian;
[initial_dictionary, param] = construct_dict(param);
alpha = 2; %gradient descent parameter, it decreases with epochs

for big_epoch = 1:10
    
    if big_epoch == 1
        learned_dictionary = initial_dictionary;
        learned_W = initial_W;
    end
    
    %% optimise with regard to x
    disp(['Epoch... ',num2str(big_epoch)]);
    x = OMP_non_normalized_atoms(learned_dictionary,param.y, param.T0);
    
    %% optimise with regard to W 
    maxEpoch = 1; %number of graph updating steps before updating sparse codes (x) again
    beta = 10^(-2); %graph sparsity penalty
    old_L = param.Laplacian;
    [param.Laplacian, learned_W] = update_graph(x, alpha, beta, maxEpoch, param, learned_W);
    [learned_dictionary, param] = construct_dict(param);
    alpha = alpha*0.985; %gradient descent decreasing
    
    % Keep track of the evolution of X and W
    X_norm_train(big_epoch) = norm(x - comp_train_X);
    norm_temp_W(big_epoch) = norm(learned_W - comp_W);
end

%% At the end of the cycle I have:
% param.alpha --> the original coefficients;
% X           --> the learned sparsity mx;
% learned_W   --> the learned W from the old D and alpha coeff;
% learned_dictionary --> the learned final dictionary;
% cpuTime     --> the final cpuTime

%constructed graph needs to be tresholded, otherwise it's too dense
%fix the number of desired edges here at nedges
nedges = 4*29;
final_Laplacian = treshold_by_edge_number(param.Laplacian, nedges);
final_W = learned_W.*(final_Laplacian~=0);


%% Estimate the final reproduction error
X_train = x;
x = OMP_non_normalized_atoms(learned_dictionary,TestSignal, param.T0);
errorTesting_Pol = sqrt(norm(TestSignal - learned_dictionary*x,'fro')^2/size(TestSignal,2));
disp(['The total representation error of the testing signals is: ',num2str(errorTesting_Pol)]);

%% Last eigenDecomposition, needed to compare the norm of the lambdas

[param.eigenMat, param.eigenVal] = eig(final_Laplacian);
[param.lambda_sym,index_sym] = sort(diag(param.eigenVal));

%% Compute the l-2 norms

X_norm_test = norm(x - comp_X);
total_X = [X_train x];
if flag == 4
    total_X_norm = norm(total_X - [comp_train_X comp_X]);
    X_norm_train = X_norm_train';
else
    X_norm_train = 'Not estimated, try with the Heat kernel dataset';
    total_X_norm = 'Not estimated, try with the Heat kernel dataset';
end
W_norm = norm(comp_W - learned_W); %Normal norm
W_norm_thr = norm(comp_W - final_W); %Normal norm of the thresholded adjacency matrix
norm_initial_W = norm(initial_W - comp_W);

%% Save results to file

% The norms
norm_temp_W = norm_temp_W';
filename = [path,num2str(ds_name),'\Norms_',num2str(ds_name),'.mat'];
save(filename,'W_norm_thr','W_norm','X_norm_train','norm_temp_W','X_norm_test','norm_initial_W','total_X_norm');

% The Output data
filename = [path,num2str(ds_name),'\Output_',num2str(ds_name),'.mat'];
learned_eigenVal = param.lambda_sym;
save(filename,'ds','learned_dictionary','learned_W','final_W','X','learned_eigenVal','errorTesting_Pol');

%% Verify the results with the precision recall function
learned_L = diag(sum(learned_W,2)) - learned_W;
learned_Laplacian = (diag(sum(learned_W,2)))^(-1/2)*learned_L*(diag(sum(learned_W,2)))^(-1/2);

comp_L = diag(sum(comp_W,2)) - comp_W;
comp_Laplacian = (diag(sum(comp_W,2)))^(-1/2)*comp_L*(diag(sum(comp_W,2)))^(-1/2);

[optPrec, optRec, opt_Lapl] = precisionRecall(comp_Laplacian, learned_Laplacian);
filename = [path,num2str(ds_name),'\ouput_PrecisionRecall_',num2str(ds_name),'.mat'];
save(filename,'opt_Lapl','optPrec','optRec');
