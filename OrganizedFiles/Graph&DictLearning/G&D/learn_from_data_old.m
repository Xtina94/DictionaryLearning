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
        X = TrainSignal;
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
        X = TrainSignal;
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
        X = TrainSignal;
        K = 15;
        param.S = 2;  % number of subdictionaries 
        param.epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
        param.N = 29; % number of nodes in the graph
        ds = 'Dataset used: data from Uber';
        ds_name = 'Uber';
        param.percentage = 8;
        param.thresh = param.percentage+6;
end

param.N = size(X,1); % number of nodes in the graph
param.J = param.N * param.S; % total number of atoms 
param.K = K*ones(1,param.S); % polynomial degree of each subdictionary
param.c = 1; % spectral control parameters
param.epsilon = 0.05;%0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
param.mu = 1;%1e-2; % polynomial regularizer paremeter
param.y = X; %signals
param.y_size = size(param.y,2);
param.T0 = 4; %sparsity level (# of atoms in each signals representation)

%% generate dictionary polynomial coefficients from heat kernel

% % % for i = 1:param.S
% % %     if mod(i,2) ~= 0
% % %         param.t(i) = 2; %heat kernel coefficients
% % %     else
% % %         param.t(i) = 1; % Inverse of the heat kernel coefficients
% % %     end
% % % end
% % % param.alpha = generate_coefficients(param);
% % % disp(param.alpha);

for i = 1:param.S
    param.alpha{i} = comp_alpha((i-1)*(K+1) + 1:i*(K+1),1);
end

%% initialise learned data

[param.Laplacian, learned_W] = init_by_weight(param.N);
[learned_dictionary, param] = construct_dict(param);
alpha = 2; %gradient descent parameter, it decreases with epochs

for big_epoch = 1:5
    %% optimise with regard to x
    disp(['Epoch... ',num2str(big_epoch)]);
    x = OMP_non_normalized_atoms(learned_dictionary,param.y, param.T0);
    
    %% optimise with regard to W 
    maxEpoch = 1; %number of graph updating steps before updating sparse codes (x) again
    beta = 10^(-2); %graph sparsity penalty
    old_L = param.Laplacian;
    [param.Laplacian, learned_W] = update_graph(x, alpha, beta, maxEpoch, param,learned_W, learned_W);
    [learned_dictionary, param] = construct_dict(param);
    alpha = alpha*0.985; %gradient descent decreasing
end

%% Estimate the final reproduction error

x = OMP_non_normalized_atoms(learned_dictionary,TestSignal, param.T0);
errorTesting_Pol = sqrt(norm(TestSignal - learned_dictionary*x,'fro')^2/size(TestSignal,2));
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
    alpha_norm = 'is 0 since here we are learning only the graph'; %norm(comp_alpha - output_Pol.alpha(1:(param.S - 1)*(degree + 1)));
    X_norm = norm(comp_X - x);
    D_norm = norm(comp_D - learned_dictionary);
    W_norm = norm(comp_W - learned_W);
    W_norm_thr = norm(comp_W - final_W); %The thresholded adjacency matrix
    
    %% Compute the average CPU_time
    
% % %     index_cpu = find(cpuTime);
% % %     my_cpu = zeros(length(index_cpu));
% % %     for i = 1:length(cpu_index)
% % %         my_cpu(i) = cpuTime(index_cpu(i));
% % %     end       
% % %     avgCPU = mean(my_cpu);

% DA ELIMINARE POI
avgCPU = 0;
    
    %% fix the last data
    

% % %     
% % %     alpha_coeff = zeros(K+1,2);
% % %     for i = 1:param.S
% % %         alpha_coeff(:,i) = param.alpha{i};
% % %     end

    %% Save the results to file
    
% % %     % The kernels plots    
% % %     figure('Name','Final kernels')
% % %     hold on
% % %     for s = 1 : param.S
% % %         plot(param.lambda_sym(2:length(param.lambda_sym)),g_ker(2:length(param.lambda_sym),s));
% % %     end
% % %     hold off
% % %     
% % %     filename = [path,'FinalKernels_plot.png'];
% % %     saveas(gcf,filename);

    % The norms
    filename = [path,'Norms.mat'];
    save(filename,'lambda_norm','alpha_norm','X_norm','D_norm','W_norm');
    
    % The Output data
    filename = [path,'Output.mat'];
% % %     learned_alpha = output_Pol.alpha;
    learned_eigenVal = param.lambda_sym;
    save(filename,'ds','learned_dictionary','learned_W','final_W','x','learned_eigenVal','errorTesting_Pol','avgCPU');

    %% Verifying the results with the precision recall function
% % %     learned_Laplacian = param.Laplacian;
% % %     [optPrec, optRec, opt_Lapl] = precisionRecall(true_Laplacian, learned_Laplacian);
% % %     filename = [path,'ouput_PrecisionRecall_attempt',num2str(attempt_index),'.mat'];
% % %     save(filename,'opt_Lapl','optPrec','optRec');
