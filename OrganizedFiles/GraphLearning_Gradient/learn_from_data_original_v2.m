%%
clear all
close all
addpath(genpath(pwd))

flag = 4;
switch flag
    case 1
        load DataSetDorina.mat
        load ComparisonDorina.mat
        X = TrainSignal;
        ds_name = 'Dorina';
        deg = 20;
        param.S = 4;  % number of subdictionaries 
    case 2
        load DataSetUber.mat
        load ComparisonUber.mat
        X = TrainSignal;
        ds_name = 'Uber';
        deg = 15;
        param.S = 2;  % number of subdictionaries 
    case 3
        load DataSetDoubleHeat.mat
        load ComparisonDoubleHeat.mat
        X = TrainSignal;
        ds_name = 'DoubleHeat';
        deg = 15;
        param.S = 2;  % number of subdictionaries 
    case 4
        load DataSetHeat30.mat
        load ComparisonHeat30.mat
        X = TrainSignal;
        ds_name = 'Heat';
        deg = 15;
        param.S = 1;  % number of subdictionaries 
end     

path = ['C:\Users\Cristina\Documents\GitHub\GraphLearningSparsityPriors\Results\05.07.18\',num2str(ds_name)];
param.N = size(X,1); % number of nodes in the graph
param.K = deg*ones(1,param.S);
param.J = param.N * param.S; % total number of atoms 
param.c = 1; % spectral control parameters
param.epsilon = 0.05;%0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
param.mu = 1;%1e-2; % polynomial regularizer paremeter
param.y = X; %signals
param.y_size = size(param.y,2);
param.conn = 5;

%% generate dictionary polynomial coefficients from heat kernel
% % % param.t(1) = 2; %heat kernel coefficients
% % % param.t(2) = 1; %this heat kernel will be inverted to cover high frequency components
% % % param.alpha = generate_coefficients(param);
K = max(param.K);
for i = 1:param.S
    param.alpha{i} = comp_alpha((K+1)*(i-1) + 1:(K+1)*i);
end
disp(param.alpha);

%% PCA over the signal
% [pca_X, score] = pca(X'); %Each column contains values for one principal component


%% initialise learned data
param.T0 = 4; %sparsity level (# of atoms in each signals representation)
[param.Laplacian, learned_W] = init_by_weight(param.N);
[learned_dictionary, param] = construct_dict(param);

%% Select the source nodes 

% compute the "Connectiveness coefficient"
n_edg = zeros(30,1);
for i = 1:size(X,1)
    n_edg(i) = length(find(comp_W(i,:)));
end
rw = diag(comp_D)./n_edg;

% Select the param.conn nodes with the highest conncectiveness coefficient
% rw_mx = [1:length(rw); rw'];
rw_sorted = sort(rw,'descend');
for i = 1:param.N/param.conn
    rw_selected(i) = rw_sorted((i-1)*param.conn+1:i*param.conn);
end
param.sources_cell = zeros(1,param.conn);
for i = 1:param.conn
    param.sources_cell{i} = find(rw == rw_selected(i));
end

D_tot = zeros(param.N);
param.Laplacian_big = zeros(param.N);
%% Make the graph sparser following the param.sources
for n_subgraphs = 1:param.N/param.conn
    %PCA over the weight matrix
    
    param.sources = param.sourcesd_cell{n_subgraphs};
    
    learned_W_small = zeros(size(learned_W,1));
    learned_W_small(param.sources,:) = learned_W(param.sources,:);
    learned_W_small(:,param.sources) = learned_W(:,param.sources);
    param.Laplacian_small = zeros(size(param.Laplacian,1));
    %  param.Laplacian_small(param.sources,:) = param.Laplacian(param.sources,:);
    param.Laplacian_small(:,param.sources) = param.Laplacian(:,param.sources);
    param.N_small = param.N;
    param.y_small = param.y;
    TestSignal_small = TestSignal;

    %% Estimate the reduced dictionary D_small
    [D_small, param] = construct_dict_small(param);
    alpha = 5; %gradient descent parameter, it decreases with epochs

    %% Graph learning algorithm

    for big_epoch = 1:250
        param.testV = big_epoch;
        %% optimise with regard to x
        disp(['Epoch... ',num2str(big_epoch)]);
        x_small = OMP_non_normalized_atoms_small(D_small,param.y_small, param.T0);
        x = OMP_non_normalized_atoms(learned_dictionary,param.y, param.T0);

        %% optimise with regard to W 
        maxEpoch = 1; %number of graph updating steps before updating sparse codes (x) again
        beta = 10^(-2); %graph sparsity penalty
        old_L = param.Laplacian;
        [param.Laplacian_small, learned_W_small] = update_graph_small(x_small, alpha, beta, maxEpoch, param,learned_W_small, learned_W_small);
        [D_small, param] = construct_dict_small(param);

        [param.Laplacian, learned_W] = update_graph_original(x, alpha, beta, maxEpoch, param,learned_W, learned_W);
        [learned_dictionary, param] = construct_dict(param);

        alpha = alpha*0.985; %gradient descent decreasing
    end
    
    D_tot = D_tot + D_small;
    param.Laplacian_big = param.Laplacian_big + param.Laplacian_small;
    
    for i = 1:param.N
        param.Laplacian_big
end
%%
%constructed graph needs to be tresholded, otherwise it's too dense
%fix the number of desired edges here at nedges
nedges = length(find(comp_W))/2;
final_Laplacian = treshold_by_edge_number(param.Laplacian, nedges);
final_W = learned_W.*(final_Laplacian~=0);

CoefMatrix_Pol_small = OMP_non_normalized_atoms(D_small,TestSignal_small, param.T0);
errorTesting_Pol_small = sqrt(norm(TestSignal_small - D_small*CoefMatrix_Pol_small,'fro')^2/size(TestSignal_small,2));

tot_X_small = [x_small CoefMatrix_Pol_small];
norm_X_small = norm(tot_X_small);

CoefMatrix_Pol = OMP_non_normalized_atoms(learned_dictionary,TestSignal, param.T0);
errorTesting_Pol = sqrt(norm(TestSignal - learned_dictionary*CoefMatrix_Pol,'fro')^2/size(TestSignal,2));

tot_X = [x CoefMatrix_Pol];
norm_X = norm(tot_X);

%% Reconstruct the big graph from the learned small graph
learned_W_big = comp_W;
learned_W_big(param.sources,:) = learned_W_small(param.sources,:);
learned_W_big(:,param.sources) = learned_W_small(:,param.sources);

L_big = diag(sum(learned_W_big,2)) - learned_W_big;
Laplacian_big = (diag(sum(learned_W_big,2)))^(-1/2)*L_big*(diag(sum(learned_W_big,2)))^(-1/2);

%% Verify the results with the precision recall function
% comp_L = diag(sum(comp_W,2)) - comp_W;
% comp_Laplacian = (diag(sum(comp_W,2)))^(-1/2)*comp_L*(diag(sum(comp_W,2)))^(-1/2);

[optPrec, optRec, opt_Lapl] = precisionRecall(comp_Laplacian, param.Laplacian);
[optPrec_small, optRec_small, opt_Lapl_small] = precisionRecall(comp_Laplacian, Laplacian_big);

%% Save results
filename = [num2str(path),'\Output_Norm_PrecRec.mat'];
save(filename,'optPrec','optRec','optPrec_small','optRec_small','CoefMatrix_Pol','errorTesting_Pol','norm_X');