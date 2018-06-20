clear all
close all

%% Adding the paths
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\Optimizers'); %Folder conatining the yalmip tools
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\Comparison_datasets\'); %Folder containing the copmarison datasets
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\'); %Folder containing the training and verification dataset
path = 'C:\Users\Cristina\Documents\GitHub\OrganizedFiles\Graph&DictLearning\ThroughConstraints\Results\'; %Folder containing the results to save

%% Loaging the required dataset
flag = 2;
switch flag
    case 1
        load ComparisonDorina.mat
        ds = 'Dataset used: Synthetic data from Dorina';
        load DataSetDorina.mat
    case 2
        load ComparisonLF.mat
        ds = 'Dataset used: data from Cristina';
        load DataSetLF.mat
    case 3
        load ComparisonUber.mat
        ds = 'Dataset used: data from Uber';
        load DataSetUber.mat        
end

%% Set the parameters

switch flag
    case 1 %Dorina
        param.S = 2;  % number of subdictionaries 
        param.epsilon = 0.55; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 20;
        param.percentage = 12;
    case 2 %Cristina
        param.S = 2;  % number of subdictionaries 
        param.epsilon = 0.55; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
        param.percentage = 8;
    case 3 %Uber
        param.S = 2;  % number of subdictionaries
        param.epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
        param.percentage = 8;
end

%% Common parameters
X = TrainSignal;
param.N = size(X,1); % number of nodes in the graph
param.J = param.N * param.S; % total number of atoms
param.K = degree*ones(1,param.S);
K = max(param.K);
param.c = 1; % spectral control parameters
param.mu = 1;%1e-2; % polynomial regularizer paremeter
param.y = X; %signals
param.y_size = size(param.y,2);
param.T0 = 2; %sparsity level (# of atoms in each signals representation)
my_max = zeros(1,param.S);

L = diag(sum(W,2)) - W; % combinatorial Laplacian
true_Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian

for repetitions = 1:1
    % Fixed stuff
    [param.Laplacian, learned_W] = init_by_weight(param.N);
    alpha_gradient = 2; %gradient descent parameter, it decreases with epochs
    
    starting_case = 2;
    switch starting_case
        case 1
            % Start from graph learning - (generate dictionary polynomial coefficients from heat kernel)
            param.t = param.K;
            for i = 1 : param.S
                param.t(i) = param.S-(i-1); %heat kernel coefficients; this heat kernel will be inverted to cover high frequency components
            end
            
            param.alpha = generate_coefficients(param);
            
            disp(param.alpha);
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
            for k = 0 : max(param.K)
                param.Laplacian_powers{k + 1} = param.Laplacian^k;
            end
    end
    
    for big_epoch = 1:20
        %% optimise with respect to x
        disp(['Epoch... ',num2str(big_epoch)]);
        x = OMP_non_normalized_atoms(learned_dictionary,param.y, param.T0);
        
        switch starting_case
            case 1
                s = 0;
            case 2
                s = 1;
        end
        
        if mod(big_epoch + s,2) ~= 0 % optimise with respect to W
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
        else % Optimize with respect to alpha
            [temp_alpha, my_max, cpuTime] = coefficient_update_interior_point(param.y,x,param,'sdpt3',big_epoch,my_max,flag);
            for j = 1:param.S
                param.alpha{j} = temp_alpha((j-1)*(K+1)+1:j*(K+1))';
            end
            [learned_dictionary, param] = construct_dict(param);
        end
        
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
        
        param.kernel = g_ker;
        
        %% Save the last plot of the dictionary learning part with no smoothness assumptions
        if big_epoch == 6
            figure('Name','Last Test on kernel behavior')
            hold on
            for s = 1 : param.S
                plot(param.lambda_sym(2:length(param.lambda_sym)),g_ker(2:length(param.lambda_sym),s));
            end
            hold off
            
            filename = strcat(num2str(repetitions),'a.png');
            fullfile = [path,'Results\',filename];
            saveas(gcf,fullfile,'png');
        end
        
    end
    
    %% Save the final learned kernels
    
    figure('Name','Final kernels')
    hold on
    for s = 1 : param.S
        plot(param.lambda_sym(2:length(param.lambda_sym)),g_ker(2:length(param.lambda_sym),s));
    end
    hold off
    
    filename = strcat(num2str(repetitions),'b.png');
    fullfile = [path,'Results\',filename];
    saveas(gcf,fullfile,'png');
    
    %% Verify the correctness of the learned data
    
    CoefMatrix_Pol = OMP_non_normalized_atoms(learned_dictionary,TestSignal, param.T0);
    errorTesting_Pol = sqrt(norm(TestSignal - learned_dictionary*CoefMatrix_Pol,'fro')^2/size(TestSignal,2));
    disp(['The total representation error of the testing signals is: ',num2str(errorTesting_Pol)]);
    sum_kernels = sum(param.kernel,2);
    
    %constructed graph needs to be tresholded, otherwise it's too dense
    %fix the number of desired edges here at nedges
    nedges = 4*29;
    final_Laplacian = treshold_by_edge_number(param.Laplacian, nedges);
    final_W = learned_W.*(final_Laplacian~=0);
    
    %% Compute the l-2 norms
    
    lambda_norm = norm(comp_eigenVal - param.eigenVal);
    alpha_norm = norm(comp_alpha - output_Pol.alpha);
    X_norm = norm(comp_X - CoefMatrix_Pol);
    D_norm = norm(comp_D - Dictionary_Pol);
    W_norm = norm(comp_W - final_W);
    
    %% Compute the average CPU_time
    
    avgCPU = mean(cpuTime);
    
    %% Save the results to file
    
    % The norms
    filename = [path,'Norms.mat'];
    save(filename,'lambda_norm','alpha_norm','X_norm','D_norm','W_norm');
    
    % The Output data
    filename = [path,'Output.mat'];
    learned_alpha = output_Pol.alpha;
    save(filename,'ds','Dictionary_Pol','learned_alpha','CoefMatrix_Pol','errorTesting_Pol','avgCPU');
    
    % The kernels plot
    figure('Name','Final Kernels')
    hold on
    for s = 1 : param.S
        plot(param.lambda_sym,g_ker(:,s));
    end
    hold off
    
    filename = [path,'FinalKernels_plot.png'];
    saveas(gcf,filename);
    
    % The CPU time plot
    xq = 0:0.2:8;
    figure('Name','CPU time per iteration')
    vq2 = interp1(1:8,output_Pol.cpuTime,xq,'spline');
    plot(1:8,output_Pol.cpuTime,'o',xq,vq2,':.');
    xlim([0 8]);
    
    filename = [path,'AvgCPUtime_plot.png'];
    saveas(gcf,filename);


    
    %%    Save results to file
    alpha_coeff = zeros(K+1,2);
    for i = 1:param.S
        alpha_coeff(:,i) = param.alpha{i};
    end
% % %     filename = 'Output_results_Uber';
% % %     save(filename,'final_Laplacian','final_W','alpha_coeff', 'g_ker','CoefMatrix_Pol','errorTesting_Pol','TrainSignal','TestSignal','sum_kernels','learned_dictionary');
    
    filename = [path,'Results\Comparison_values_UberData'];
    save(filename,'final_W','alpha_coeff','CoefMatrix_Pol','learned_dictionary');
    
    %% Verifying the results with the precision recall function
    % Save the results in a file in which we have also the reconstruction error
    % such that we can compare it with the algo imposes the kernels' behavior
    % from the beginning
    if flag == 2
        learned_Laplacian = param.Laplacian;
        [optPrec, optRec, opt_Lapl] = precisionRecall(true_Laplacian, learned_Laplacian);
        filename = strcat(path,'Results\','ouput_PrecisionRecall',num2str(repetitions),'.mat');
        save(filename,'opt_Lapl','optPrec','optRec','errorTesting_Pol');
    end
end