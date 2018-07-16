%%
clear all
close all
addpath(genpath(pwd))

load DataSetDorina.mat
load ComparisonDorina.mat
ds_name = 'Dorina';
path = 'C:\Users\Cristina\Documents\GitHub\GraphLearningSparsityPriors\';
X = TrainSignal;

%load data in variable X here (it should be a matrix #nodes x #signals)

param.N = size(X,1); % number of nodes in the graph
param.S = 4;  % number of subdictionaries 
param.J = param.N * param.S; % total number of atoms 
param.K = [20 20 20 20]; % polynomial degree of each subdictionary
param.c = 1; % spectral control parameters
param.epsilon = 0.05;%0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
param.mu = 1;%1e-2; % polynomial regularizer paremeter
param.y = X; %signals
param.y_size = size(param.y,2);

%% generate dictionary polynomial coefficients from heat kernel
% % % param.t(1) = 2; %heat kernel coefficients
% % % param.t(2) = 1; %this heat kernel will be inverted to cover high frequency components
% % % param.alpha = generate_coefficients(param);
% % % disp(param.alpha);

for i = 1:param.S
    param.alpha{i} = comp_alpha((i-1)*(max(param.K)+1) + 1:(max(param.K)+1)*i);
end

%% initialise learned data
param.T0 = 6; %sparsity level (# of atoms in each signals representation)
[param.Laplacian, learned_W] = init_by_weight(param.N);
[learned_dictionary, param] = construct_dict(param);
alpha = 2; %gradient descent parameter, it decreases with epochs
for big_epoch = 1:20
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
end

%%
%constructed graph needs to be tresholded, otherwise it's too dense
%fix the number of desired edges here at nedges
nedges = 4*29;
final_Laplacian = treshold_by_edge_number(param.Laplacian, nedges);
final_W = learned_W.*(final_Laplacian~=0);

%% Estimate the final reproduction error
X = OMP_non_normalized_atoms(learned_dictionary,TestSignal, param.T0);
errorTesting_Pol = sqrt(norm(TestSignal - learned_dictionary*X,'fro')^2/size(TestSignal,2));
disp(['The total representation error of the testing signals is: ',num2str(errorTesting_Pol)]);

%% Save results
filename = [path,num2str(ds_name),'\Output_',num2str(ds_name),'.mat'];
learned_eigenVal = 0; %param.lambda_sym;
save(filename,'learned_dictionary','learned_W','final_W','X','learned_eigenVal','errorTesting_Pol');

%% Verify the results with the precision recall function
learned_L = diag(sum(learned_W,2)) - learned_W;
learned_Laplacian = (diag(sum(learned_W,2)))^(-1/2)*learned_L*(diag(sum(learned_W,2)))^(-1/2);

comp_L = diag(sum(comp_W,2)) - comp_W;
comp_Laplacian = (diag(sum(comp_W,2)))^(-1/2)*comp_L*(diag(sum(comp_W,2)))^(-1/2);

[optPrec, optRec, opt_Lapl] = precisionRecall(comp_Laplacian, learned_Laplacian);
filename = [path,num2str(ds_name),'\ouput_PrecisionRecall_',num2str(ds_name),'.mat'];
save(filename,'opt_Lapl','optPrec','optRec');
