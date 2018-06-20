clear all
close all

%% Adding the paths
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\Optimizers'); %Folder conatining the yalmip tools
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\Comparison_datasets\'); %Folder containing the copmarison datasets
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\'); %Folder containing the training and verification dataset
path = 'C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DictionaryLearning\AddingThirdKernel\Results\'; %Folder containing the results to save

%% Loading the required dataset
flag = 1;
switch flag
    case 1
        load ComparisonDorina.mat
        load DataSetDorina.mat
    case 2
        load ComparisonLF.mat
        load DataSetLF.mat
    case 3
        load ComparisonUber.mat
        load DataSetUber.mat
end

%% Set the parameters

switch flag
    case 1 %Dorina
        param.S = 5;  % number of subdictionaries 
        param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 20;
        param.N = 100; % number of nodes in the graph
        ds = 'Dataset used: Synthetic data from Dorina';
        ds_name = 'Dorina';
        param.percentage = 15;
        param.thresh = param.percentage+60;
    case 2 %Cristina
        param.S = 3;  % number of subdictionaries 
        param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
        param.N = 100; % number of nodes in the graph
        ds = 'Dataset used: data from Cristina';
        ds_name = 'Cristina'; 
        param.percentage = 8;
        param.thresh = param.percentage+60;
    case 3 %Uber
        param.S = 3;
        param.epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
        param.N = 29; % number of nodes in the graph
        ds = 'Dataset used: data from Uber';
        ds_name = 'Uber';
        param.percentage = 8;
        param.thresh = param.percentage+6;
end

% Fixed stuff
param.J = param.N * param.S; % total number of atoms 
param.K = degree*ones(1,param.S);
param.y = TrainSignal;
K = max(param.K);
param.T0 = 4; % sparsity level in the training phase
param.c = 1; % spectral control parameters
param.mu = 1e-2; % polynomial regularizer paremeter
[param.Laplacian, learned_W] = init_by_weight(param.N);
alpha_gradient = 2; %gradient descent parameter, it decreases with epochs
attempt_n = 1;

for attempt_index = 1:attempt_n    
    starting_case = 2;
    switch starting_case
        case 1
            % Start from graph learning - (generate dictionary polynomial coefficients from heat kernel)
            param.t = param.K;
            for i = 1 : param.S
                param.t(i) = param.S-(i-1); %heat kernel coefficients; this heat kernel will be inverted to cover high frequency components
            end
            
            param.alpha = generate_coefficients(param);
            [learned_dictionary, param] = construct_dict(param);
        case 2
            % Start from Dictionary learning - (Initialize the dictionary and the alpha coefficients' structure)
            [param.eigenMat, param.eigenVal] = eig(param.Laplacian); % eigendecomposition of the normalized Laplacian
            [learned_dictionary] = initialize_dictionary(param);
            [param.lambda_sym,index_sym] = sort(diag(param.eigenVal)); % sort the eigenvalues of the normalized Laplacian in descending order
            [param.beta_coefficients, param.rts] = retrieve_betas(param);
            param.lambda_power_matrix = zeros(param.N,K+1);
            for i = 0:K
                param.lambda_power_matrix(:,i+1) = param.lambda_sym.^i;
            end
            for k = 0:K
                param.Laplacian_powers{k + 1} = param.Laplacian^k;
            end
    end
    
    cpuTime = zeros(1,20);
    for big_epoch = 1:20
        param.numIteration = big_epoch;
        %% optimise with respect to x
        disp(['Epoch... ',num2str(big_epoch)]);
        x = OMP_non_normalized_atoms(learned_dictionary,param.y, param.T0);
        switch starting_case
            case 1
                s = 0;
            case 2
                s = 1;
        end 
        %% Optimize with respect to W
        if mod(big_epoch + s,2) ~= 0
            maxEpoch = 1; %number of graph updating steps before updating sparse codes (x) again
            beta = 10^(-2); %graph sparsity penalty
            old_L = param.Laplacian;
            [param.Laplacian, learned_W] = update_graph(x, alpha_gradient, beta, maxEpoch, param,learned_W, learned_W);
            [learned_dictionary, param] = construct_dict(param);
            alpha_gradient = alpha_gradient*0.985; %gradient descent decreasing         
            [param.eigenMat, param.eigenVal] = eig(param.Laplacian); % eigendecomposition of the normalized Laplacian
            [param.lambda_sym,index_sym] = sort(diag(param.eigenVal)); % sort the eigenvalues of the normalized Laplacian in descending order
            param.lambda_power_matrix = zeros(param.N,K+1);
            for i = 0:K
                param.lambda_power_matrix(:,i+1) = param.lambda_sym.^i;
            end
            for k = 0 : max(param.K)
                param.Laplacian_powers{k + 1} = param.Laplacian^k;
            end
        %% Optimize with respect to alpha
        else
            [temp_alpha, cpuTm] = coefficient_update_interior_point(TrainSignal,x,param,'sdpt3');
            cpuTime(big_epoch) = cpuTm;
            for j = 1:param.S
                param.alpha{j} = temp_alpha((j-1)*(K+1)+1:j*(K+1))';
            end
            [learned_dictionary, param] = construct_dict(param);
        end
        
        %% construct the kernels polynomial 
        g_ker = zeros(param.N, param.S);
        for i = 1 : param.S
            for n = 1 : param.N
                p = 0;
                for l = 0 : param.K(i)
                    p = p +  param.alpha{i}(l+1)*param.lambda_power_matrix(n,l + 1);
                end
                g_ker(n,i) = p;
            end
        end
        
        if big_epoch == 6
            figure('Name','Kernels learned without constraints')
            hold on
            for s = 1 : param.S
                plot(param.lambda_sym(2:length(param.lambda_sym)),g_ker(2:length(param.lambda_sym),s));
            end
            hold off
            
        filename = [path,'Intermediate_kernel_plot.png'];
        saveas(gcf,filename);
        end
        
    end
    
    %% Compute the l-2 norms
    
    lambda_norm = 'is 0 since here we are learning only the kernels'; %norm(comp_eigenVal - eigenVal);
    alpha_norm = norm(comp_alpha - output_Pol.alpha(1:(param.S - 1)*(degree + 1)));
    X_norm = norm(comp_X - CoefMatrix_Pol(1:(param.S - 1)*param.N,:));
    D_norm = norm(comp_D - Dictionary_Pol(:,1:(param.S - 1)*param.N));
    W_norm = 'is 0 since here we are learning only the kernels';
    
    %% Compute the average CPU_time
    
    index_cpu = find(cpuTime);
    my_cpu = zeros(length(index_cpu));
    for i = 1:length(cpu_index)
        my_cpu(i) = cpuTime(index_cpu(i));
    end       
    avgCPU = mean(my_cpu);
    
    %% fix the last data
    
    CoefMatrix_Pol = OMP_non_normalized_atoms(learned_dictionary,TestSignal, param.T0);
    errorTesting_Pol = sqrt(norm(TestSignal - learned_dictionary*CoefMatrix_Pol,'fro')^2/size(TestSignal,2));
    disp(['The total representation error of the testing signals is: ',num2str(errorTesting_Pol)]);
    sum_kernels = sum(param.kernel,2);
    
    alpha_coeff = zeros(K+1,2);
    for i = 1:param.S
        alpha_coeff(:,i) = param.alpha{i};
    end
    
    %constructed graph needs to be tresholded, otherwise it's too dense
    %fix the number of desired edges here at nedges
    nedges = 4*29;
    final_Laplacian = treshold_by_edge_number(param.Laplacian, nedges);
    final_W = learned_W.*(final_Laplacian~=0);

    %% Save the results to file
    
    % The kernels plots    
    figure('Name','Final kernels')
    hold on
    for s = 1 : param.S
        plot(param.lambda_sym(2:length(param.lambda_sym)),g_ker(2:length(param.lambda_sym),s));
    end
    hold off
    
    filename = [path,'FinalKernels_plot.png'];
    saveas(gcf,filename);

    % The norms
    filename = [path,'Norms.mat'];
    save(filename,'lambda_norm','alpha_norm','X_norm','D_norm','W_norm');
    
    % The Output data
    filename = [path,'Output.mat'];
    learned_alpha = output_Pol.alpha;
    save(filename,'ds','Dictionary_Pol','learned_alpha','CoefMatrix_Pol','errorTesting_Pol','avgCPU');

    %% Verifying the results with the precision recall function
    learned_Laplacian = param.Laplacian;
    [optPrec, optRec, opt_Lapl] = precisionRecall(true_Laplacian, learned_Laplacian);
    filename = [path,'ouput_PrecisionRecall_attempt',num2str(attempt_index),'.mat'];
    save(filename,'opt_Lapl','optPrec','optRec');
end