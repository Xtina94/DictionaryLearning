%%
clear all
close all

%% Adding the paths
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\Optimizers'); %Folder conatining the yalmip tools
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\Comparison_datasets\'); %Folder containing the copmarison datasets
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\'); %Folder containing the training and verification dataset
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\GeneratingKernels\Results'); %Folder conatining the heat kernel coefficietns

flag = 4;
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
    case 4 %Heat kernel
        Y = TrainSignal;
        K = 15;
        param.S = 1;  % number of subdictionaries 
        param.epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
        param.N = 100; % number of nodes in the graph
        ds = 'Dataset used: data from Heat kernel';
        ds_name = 'Heat';
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
param.max = 0;
path = ['C:\Users\Cristina\Documents\GitHub\OrganizedFiles\Graph&DictLearning\LearningG&D\Results\',num2str(ds_name),'\']; %Folder containing the results to save


%% Obtain the initial Laplacian and eigenValues
comp_L = diag(sum(comp_W,2)) - comp_W; % combinatorial Laplacian
comp_Laplacian = (diag(sum(comp_W,2)))^(-1/2)*comp_L*(diag(sum(comp_W,2)))^(-1/2); % normalized Laplacian
[comp_eigenMat, comp_eigenVal] = eig(comp_Laplacian);
[comp_lambdaSym,comp_indexSym] = sort(diag(comp_eigenVal));
comp_lambdaPowerMx(:,2) = comp_lambdaSym;
for i = 1:max(param.K) + 1
    comp_lambdaPowerMx(:,i) = comp_lambdaPowerMx(:,2).^(i-1);
end

comp_ker = zeros(param.N,param.S);
for i = 1 : param.S
    for n = 1:param.N
        comp_ker(n,i) = comp_ker(n,i) + comp_lambdaPowerMx(n,:)*comp_alpha(:,i);
    end
end

%% Initialize the unknown quantities

% % % %% generate dictionary polynomial coefficients from heat kernel if I don't already have them
% % % 
% % % if flag == 4
% % %     param.alpha = LF_heatKernel;
% % % else
% % %     for i = 1:param.S
% % %         if mod(i,2) ~= 0
% % %             param.t(i) = 2; %heat kernel coefficients
% % %         else
% % %             param.t(i) = 1; %Inverse of the heat kernel coefficients
% % %         end
% % %     end
% % %     param.alpha = generate_coefficients(param);
% % %     disp(param.alpha);
% % %     
% % %     for i = 1:param.S
% % %         param.alpha{i} = comp_alpha((i-1)*(K+1) + 1:i*(K+1),1);
% % %     end
% % % end

%% Initialise W: 

uniform_values = unifrnd(0,1,[1,param.N]);
sigma = 0.2;
if flag == 4
    [initial_W,L] = random_geometric(sigma,param.N,uniform_values,0.6);
    param.Laplacian = L;
else
    [param.Laplacian, initial_W] = init_by_weight(param.N);
end

[param.eigenVal, param.eigenMat] = eig(param.Laplacian);
param.lambda_sym = sort(diag(param.eigenVal));
param.lambda_power_matrix = zeros(param.N,max(param.K) + 1);
for i = 1:max(param.K)+1
    param.Laplacian_powers{i} = param.Laplacian^(i-1);
    param.lambda_power_matrix(:,i) = param.lambda_sym.^(i-1);
end

norm_initial_W = norm(initial_W - comp_W);

%% Initialize the dicitonary and the eigenValues
[initial_dictionary(:,1 : param.J)] = initialize_dictionary(param);

%% Initialize the other standard values
grad_desc = 2; %gradient descent parameter, it decreases with epochs
g_ker = zeros(param.N, param.S);

for big_epoch = 1:10      
    
    param.iterN = big_epoch;
    if big_epoch == 1
        learned_dictionary = initial_dictionary;
        learned_W = initial_W;
    end
    
    %% optimise with regard to x
    disp(['Epoch... ',num2str(big_epoch)]);
    X = OMP_non_normalized_atoms(learned_dictionary,param.y, param.T0);
    
    % Keep track of the evolution of X
    X_norm_train(big_epoch) = norm(X - comp_train_X);
    
    if mod(big_epoch,2) == 0
        %% optimise with respect to W
        maxEpoch = 1; %number of graph updating steps before updating sparse codes (x) again
        beta = 10^(-2); %graph sparsity penalty
        old_L = param.Laplacian;
        [param.Laplacian, learned_W] = update_graph(X, grad_desc, beta, maxEpoch, param, learned_dictionary, learned_W);
        
        % Re-obtain the dictionary etc
        [learned_dictionary, param] = construct_dict(param);
        grad_desc = grad_desc*0.985; %gradient descent decreasing
        
        % Keep track of the evolution of X
        norm_temp_W(big_epoch) = norm(learned_W - comp_W);        
    else
        %% Optimize with respect to alpha
        [param,cpuTm] = coefficient_update_interior_point(Y,X,param,'sdpt3',g_ker);
        cpuTime((big_epoch + 1)/2) = cpuTm;
        
        % Re-obtain the Laplacian, the dictionary etc
        learned_L = diag(sum(learned_W,2)) - learned_W;
        param.Laplacian = (diag(sum(learned_W,2)))^(-1/2)*learned_L*(diag(sum(learned_W,2)))^(-1/2);
        [learned_dictionary, param] = construct_dict(param);
    end
end

%% At the end of the cycle I have:
% param.alpha --> the learned coefficients;
% X           --> the learned sparsity mx;
% learned_W   --> the learned W from the old D and alpha coeff;
% learned_dictionary --> the learned final dictionary;
% cpuTime     --> the final cpuTime

%% Estimate the final reproduction error
X_train = X;
X = OMP_non_normalized_atoms(learned_dictionary,TestSignal, param.T0);
errorTesting_Pol = sqrt(norm(TestSignal - learned_dictionary*X,'fro')^2/size(TestSignal,2));
disp(['The total representation error of the testing signals is: ',num2str(errorTesting_Pol)]);

%%
%constructed graph needs to be tresholded, otherwise it's too dense
%fix the number of desired edges here at nedges
nedges = 4*param.N;
final_Laplacian = treshold_by_edge_number(param.Laplacian, nedges);
final_W = learned_W.*(final_Laplacian~=0);

%% Last eigenDecomposition, needed to compare the norm of the lambdas

[param.eigenMat, param.eigenVal] = eig(final_Laplacian);
[param.lambda_sym,index_sym] = sort(diag(param.eigenVal));

%% Represent the kernels

param.lambda_power_matrix(:,2) = param.lambda_sym;
for i = 1:max(param.K) + 1
    param.lambda_power_matrix(:,i) = param.lambda_power_matrix(:,2).^(i-1);
end

g_ker = zeros(param.N,param.S);
for i = 1 : param.S
    for n = 1:param.N
        g_ker = g_ker + param.lambda_power_matrix(n,:)*param.alpha(:,i);
    end
end

figure('Name','Comparison between the Kernels')
    subplot(1,2,1)
    title('Original kernels');
    hold on
    for s = 1 : param.S
        plot(comp_lambdaSym,comp_ker(:,s));
    end
    hold off
    subplot(1,2,2)
    title('learned kernels');
    hold on
    for s = 1 : param.S
        plot(param.lambda_sym,g_ker(:,s));
    end
    hold off

%% Compute the l-2 norms

X_norm_test = norm(X - comp_X);
total_X = [X_train X];
total_X_norm = norm(total_X - [comp_train_X comp_X]);
W_norm = norm(comp_W - learned_W); %Normal norm
W_norm_thr = norm(comp_W - final_W); %Normal norm of the thresholded adjacency matrix
% % % norm_temp_X(big_epoch + 1) = norm(X - temp_X);
% % % W_norm_FRO = sqrt(norm(comp_W - learned_W,'fro')^2/size(comp_W,2)); %Frobenius norm
% % % W_norm_thr_FRO = sqrt(norm(comp_W - final_W,'fro')^2/size(comp_W,2)); %Frobenius norm of the thresholded adjacency matrix

%% Graphically represent the behavior od the learned entities

figure('name','Behavior of the X (blue line) and the W (orange line)')
hold on
plot(1:10,X_norm_train)
plot(1:10,norm_temp_W)
hold off

%% Save the results to file

% The norms
norm_temp_W = norm_temp_W';
X_norm_train = X_norm_train';
filename = [path,num2str(ds_name),'\Norms_',num2str(ds_name),'.mat'];
save(filename,'W_norm_thr','W_norm','X_norm_train','norm_temp_W','X_norm_test','norm_initial_W','total_X_norm');

% The Output data
filename = [path,num2str(ds_name),'\Output_',num2str(ds_name),'.mat'];
learned_eigenVal = param.lambda_sym;
save(filename,'ds','learned_dictionary','learned_W','final_W','X','learned_eigenVal','errorTesting_Pol');

%% Verify the results with the precision recall function
learned_Laplacian = final_Laplacian;
[optPrec, optRec, opt_Lapl] = precisionRecall(comp_Laplacian, learned_Laplacian);
filename = [path,num2str(ds_name),'\ouput_PrecisionRecall_',num2str(ds_name),'.mat'];
save(filename,'opt_Lapl','optPrec','optRec');



