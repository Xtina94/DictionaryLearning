 clear all
close all

%% Adding the paths
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\Optimizers'); %Folder conatining the yalmip tools
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\Comparison_datasets\'); %Folder containing the comparison datasets
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\'); %Folder containing the training and verification dataset
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\GeneratingKernels\Results'); %Folder conatining the heat kernel coefficients
path = 'C:\Users\Cristina\Documents\GitHub\OrganizedFiles\Graph&DictLearning\G&D_fromGraphCorrect\Results\13.07.18\'; %Folder containing the results to save

flag = 7;

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
    case 4 %1 Heat kernel with 30 nodes
        load DataSetHeat30.mat;
        load ComparisonHeat30.mat;
%         load prova.mat;
%         comp_alpha = prova{1};
    case 5
        load DataSetDoubleHeat.mat;
        load ComparisonDoubleHeat.mat;
        load HeatKernel.mat;
    case 6
        load DataSetDoubleInvHeat.mat;
        load ComparisonDoubleInvHeat.mat;
%         load provaDoubleSmth.mat;
    case 7
        load DataSetDorinaLF.mat;
        load ComparisonDorinaLF.mat;
end

switch flag
    case 1 %Dorina
        Y = TrainSignal;
        K = 20;
        param.S = 4;  % number of subdictionaries         
        param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 20;
        ds = 'Dataset used: Synthetic data from Dorina';
        ds_name = 'Dorina';
        param.percentage = 15;
        param.thresh = param.percentage + 60;
        alpha = 20; %gradient descent parameter, it decreases with epochs
    case 2 %Uber
        Y = TrainSignal;
        K = 15;
        param.S = 2;  % number of subdictionaries 
        param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
        ds = 'Dataset used: data from Uber';
        ds_name = 'Uber';
        param.percentage = 8;
        param.thresh = param.percentage + 6;
        alpha = 5; %gradient descent parameter, it decreases with epochs
    case 3 %Cristina
        Y = TrainSignal;
        K = 15;
        param.S = 2;  % number of subdictionaries        
        param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
        ds = 'Dataset used: data from Cristina';
        ds_name = 'Cristina'; 
        param.percentage = 8;
        param.thresh = param.percentage + 6;
        alpha = 5; %gradient descent parameter, it decreases with epochs
    case 4 %Heat kernel
        Y = TrainSignal;
        K = 15;
        param.S = 1;  % number of subdictionaries 
        param.epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
        ds = 'Dataset used: data from Heat kernel';
        ds_name = 'Heat';        
        param.percentage = 8;
        param.thresh = param.percentage + 18;
        alpha = 5; %gradient descent parameter, it decreases with epochs
    case 5 %Heat kernel
        Y = TrainSignal;
        K = 15;
        param.S = 2;  % number of subdictionaries 
        param.epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
        ds = 'Dataset used: data from double Heat kernel';
        ds_name = 'DoubleHeat';        
        param.percentage = 8;
        param.thresh = param.percentage + 2;
        alpha = 15; %gradient descent parameter, it decreases with epochs
    case 6 %Double inv heat kernel
        Y = TrainSignal;
        K = 15;
        param.S = 2;  % number of subdictionaries 
        param.epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
        ds = 'Dataset used: data from double Heat kernel';
        ds_name = 'DoubleHeat';        
        param.percentage = 8;
        param.thresh = param.percentage + 18;
        alpha = 15; %gradient descent parameter, it decreases with epochs
        
        for i = 1:param.S
            comp_alpha((K+1)*(i-1) + 1:(K+1)*i) = provaDoubleSmth{i};
        end
    case 7
        Y = TrainSignal;
        K = 20;
        param.S = 1;  % number of subdictionaries         
        param.epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 20;
        ds = 'Dataset used: Synthetic data from Dorina - 1 single kernel';
        ds_name = 'DorinaLF';
        param.percentage = 15;
        param.thresh = param.percentage + 13;
        alpha = 20; %gradient descent parameter, it decreases with epochs
end

param.N = size(Y,1); % number of nodes in the graph
param.J = param.N * param.S; % total number of atoms 
param.K = K*ones(1,param.S); %[20 20 20 20]; % polynomial degree of each subdictionary
param.c = 1; % spectral control parameters
param.mu = 1;%1e-2; % polynomial regularizer paremeter
param.y = Y; %signals
param.y_size = size(param.y,2);
param.T0 = 4; %sparsity level (# of atoms in each signals representation)
param.max = 0;

%% Initialize the kernel coefficients
temp = comp_alpha;
comp_alpha = zeros(K+1,param.S);
for i = 1:param.S
    comp_alpha(:,i) = temp((K+1)*(i-1) + 1:(K+1)*i);
    param.alpha{i} = comp_alpha(:,i);
end

% % %     for i = 1:param.S
% % %         param.alpha{i} = randn(K+1,1); %comp_alpha(:,i);
% % %     end

[comp_lambdaSym,comp_indexSym] = sort(diag(comp_eigenVal));
comp_lambdaPowerMx(:,2) = comp_lambdaSym;

for i = 1:K+1
    comp_lambdaPowerMx(:,i) = comp_lambdaPowerMx(:,2).^(i-1);
    comp_Laplacian_powers{i} = comp_Laplacian^(i-1);
end

comp_ker = zeros(param.N,param.S);
for i = 1 : param.S
    for n = 1:param.N
        comp_ker(n,i) = comp_ker(n,i) + comp_lambdaPowerMx(n,:)*comp_alpha(:,i);
    end
end

%% Initialize W:

uniform_values = unifrnd(0,1,[1,param.N]);
sigma = 0.2;
[initial_W,param.Laplacian] = random_geometric(sigma,param.N,uniform_values,0.6);
% [param.Laplacian, initial_W] = init_by_weight(param.N);
initial_Laplacian = param.Laplacian;

[param.eigenMat, param.eigenVal] = eig(param.Laplacian);
[param.lambdaSym,indexSym] = sort(diag(param.eigenVal));
param.lambda_power_matrix(:,2) = param.lambdaSym;

for i = 1:max(param.K) + 1
    param.lambda_power_matrix(:,i) = param.lambda_power_matrix(:,2).^(i-1);
end

%% Initialize D:
[initial_dictionary, param] = construct_dict(param);

maxIter = 200;
X_norm_train = zeros(maxIter,1);
D_norm_train = zeros(maxIter,1);
norm_temp_W = zeros(maxIter,1);
D_diff = cell(maxIter,1);
W_vector = cell(1,200);

for big_epoch = 1:maxIter
    
    param.iterN = big_epoch;
    
    if big_epoch == 1
        learned_dictionary = initial_dictionary;
        learned_W = initial_W;
        g_ker = zeros(param.N, param.S);
    end
    
    %---------optimise with respect to X---------%
    
    disp(['Epoch... ',num2str(big_epoch)]);
    x = OMP_non_normalized_atoms(learned_dictionary,param.y, param.T0);
    
    if mod(big_epoch,5) == 0
        %------optimize with respect to alpha------%
        
        roots = retrieve_betas(param);
        [param,cpuTm] = coefficient_update_interior_point(Y,x,param,'sdpt3',roots);
%         cpuTime((big_epoch + 1)/2) = cpuTm;        
    else
        %--------optimise with respect to W--------%
        disp('Graph learning step');
        maxEpoch = 1; %number of graph updating steps before updating sparse codes (x) again
        beta = 10^(-4); %graph sparsity penalty
        old_L = param.Laplacian;
        [param.Laplacian, learned_W,param.lambda_sym] = update_graph(x, alpha, beta, maxEpoch, param, learned_W);
        alpha = alpha*0.985; %gradient descent decreasing
    end
    
    W_vector{big_epoch} = learned_W;
    
    % Re-obtain D
    [learned_dictionary, param] = construct_dict(param);
    
    % Keep track of the evolution of D
    D_norm_train_def = 'norm(learned_dictionary - comp_D)';
    D_norm_train(big_epoch) = norm(learned_dictionary - comp_D);
    
    % Analyse the structural difference between learned Dictionaries
    D_diff{big_epoch} = (learned_dictionary - comp_D);
    
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
nedges = length(find(comp_W))/2;
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
total_X_norm = norm(total_X - [comp_train_X comp_X]);
% X_norm_train = X_norm_train';
W_norm = norm(comp_W - learned_W); %Normal norm
W_norm_thr = norm(comp_W - final_W); %Normal norm of the thresholded adjacency matrix
norm_initial_W = norm(initial_W - comp_W);
norm_init_D = norm(initial_dictionary - comp_D);

alpha_norm = zeros(param.S,1);
alpha_diff = zeros(K+1,param.S);
for i = 1:param.S
    alpha_norm(i,1) = norm(comp_alpha(:,i) - param.alpha{i});
    alpha_diff(:,i) = (comp_alpha(:,i) - param.alpha{i});
end

%% Represent the difference in the alphas

figure('Name','The different behavior of the alpha coefficients')
for j = 1:param.S
    subplot(1,param.S,j)
    title(['Comp ker ',num2str(j),'  VS Learned ker ',num2str(j)]);
    hold on
    stem(comp_alpha(:,j))
    stem(param.alpha{j})
    hold off
    legend('comparison kernel','learned kernel');
end
filename = [path,num2str(ds_name),'\AlphaCoeff_comparison','.png'];
saveas(gcf,filename);

%% Write down the definition of the norms for better clearance
norm_initial_W_def = 'norm(initial_W - comp_W)';
norm_temp_W_def = 'norm(learned_W - comp_W)';
X_norm_train_def = 'norm(X - comp_train_X)';
W_norm_thr_def = 'norm(comp_W - final_W) --> Normal norm of the thresholded adjacency matrix';
W_norm_def = 'norm(comp_W - learned_W)';
total_X_norm_def = 'norm(total_X - [comp_train_X comp_X])';
X_norm_test_def = 'norm(X - comp_X)';
norm_init_D_def = 'norm(initial_dictionary - comp_D)';
alpha_norm_def = 'norm(comp_alpha(:,i) - param.alpha{i});';
alpha_diff_def = '(comp_alpha(:,i) - param.alpha{i});';

%% Graphically represent the behavior od the learned entities

figure('name','Behavior of the X_norm_train (blue line) and the D_norm_train (orange line)')
hold on
grid on
plot(1:maxIter,X_norm_train);
plot(1:maxIter,D_norm_train);
hold off

filename = [path,num2str(ds_name),'\behaviorX_','.fig'];
saveas(gcf,filename);

na = zeros(maxIter,1);
ne = zeros(maxIter,1);
ni = zeros(maxIter,1);
for i = 1:maxIter
    na(i) = norm(W_vector{i});
    ne(i) = norm_temp_W(i);
    ni(i) = norm(comp_W);
end

figure('Name','Behavior of the W')
hold on
plot(ni)
plot(na)
plot(ne)
hold off
legend('comp\_W norm','W\_vect norm','temp\_W norm');

filename = [path,num2str(ds_name),'\behaviorW_','.fig'];
saveas(gcf,filename);

%% Represent the kernels

param.lambda_power_matrix(:,2) = param.lambda_sym;
for i = 1:max(param.K) + 1
    param.lambda_power_matrix(:,i) = param.lambda_power_matrix(:,2).^(i-1);
end

for i = 1 : param.S
    for n = 1:param.N
        g_ker(n,i) = param.lambda_power_matrix(n,:)*param.alpha{i};
    end
end

figure('Name','Comparison between the Kernels')
subplot(2,1,1)
title('Original kernels');
hold on
for s = 1 : param.S
    plot(comp_lambdaSym,comp_ker(:,s));
end
hold off
subplot(2,1,2)
title('learned kernels');
hold on
for s = 1 : param.S
    plot(param.lambda_sym(1:length(param.lambda_sym)-1),g_ker(1:length(g_ker)-1,s));
end
hold off

%% Save results to file

% The kernels
filename = [path,num2str(ds_name),'\Comp_kernels_','.png'];
saveas(gcf,filename);

% The norms
norm_temp_W = norm_temp_W';
filename = [path,num2str(ds_name),'\Norms_',num2str(ds_name),'.mat'];
save(filename,'W_norm_thr','W_norm','X_norm_train','norm_temp_W','X_norm_test','norm_initial_W','total_X_norm','alpha_norm','alpha_norm_def','alpha_diff','alpha_diff_def');

% The Output data
filename = [path,num2str(ds_name),'\Output_',num2str(ds_name),'.mat'];
learned_eigenVal = param.lambda_sym;
save(filename,'ds','learned_dictionary','learned_W','final_W','Y','learned_eigenVal','errorTesting_Pol');

%% Verify the results with the precision recall function
learned_L = diag(sum(learned_W,2)) - learned_W;
learned_Laplacian = (diag(sum(learned_W,2)))^(-1/2)*learned_L*(diag(sum(learned_W,2)))^(-1/2);
% % % learned_Laplacian = final_Laplacian;
% % % comp_L = diag(sum(comp_W,2)) - comp_W;
% % % comp_Laplacian = (diag(sum(comp_W,2)))^(-1/2)*comp_L*(diag(sum(comp_W,2)))^(-1/2);

[optPrec, optRec, opt_Lapl] = precisionRecall(comp_Laplacian, learned_Laplacian);
filename = [path,num2str(ds_name),'\ouput_PrecisionRecall_',num2str(ds_name),'.mat'];
save(filename,'opt_Lapl','optPrec','optRec');
