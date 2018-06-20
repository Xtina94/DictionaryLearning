clear all
close all
addpath(genpath(pwd))
path = 'C:\Users\Cristina\Documents\GitHub\DictionaryLearning\LearningPart\Analyzing the l2norm\';
addpath(strcat(path,'DataSets'))

% load data in variable X here (it should be a matrix #nodes x #signals)
for repetitions = 1:1
    attempt_n = repetitions;
    flag = 3;
    switch flag
        case 1
            % For synthetic data
            load testdata.mat
            param.S = 2;  % number of subdictionaries
            param.K = [20 20]; % polynomial degree of each subdictionary
            param.percentage = 12;
            param.epsilon = 0.55; %0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
            L = diag(sum(W,2)) - W; % combinatorial Laplacian
            true_Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian
            
            load Comparison_values_testData.mat
            reference_kernels = alpha;
%             reference_sparsity = X;
            reference_D = my_D;
            reference_W = W;
        case 2
            % For Uber
            param.S = 2;  % number of subdictionaries
            param.K = [15 15]; % polynomial degree of each subdictionary
            param.percentage = 8;
            param.epsilon = 0.2; %0.02; % we assume that epsilon_1 = epsilon_2 = epsilon

            load UberData.mat
            TestSignal = X(:,91:110);
            TrainSignal = X(:,1:90);
            L = diag(sum(learned_W,2)) - learned_W; % combinatorial Laplacian
            true_Laplacian = (diag(sum(learned_W,2)))^(-1/2)*L*(diag(sum(learned_W,2)))^(-1/2); % normalized Laplacian
            
            load Comparison_values_UberData.mat
            reference_kernels = alpha;
%             reference_sparsity = X;
            reference_D = my_D;
            reference_W = W;
        case 3
            load DataSet_kernels_LF_v2.mat
            TestSignal = Y(:,901:1000);
            TrainSignal = Y(:,1:900);
            param.S = 2;  % number of subdictionaries
            param.K = [15 15]; % polynomial degree of each subdictionary
            param.percentage = 8;
            param.epsilon = 0.55; %0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
            
            L = diag(sum(W,2)) - W; % combinatorial Laplacian
            true_Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian
     
            load Comparison_values_LF_v2.mat
            reference_kernels = alpha;
            reference_sparsity = X;
            reference_D = my_D;
            reference_W = W;
    end
    
    % Common
    X = TrainSignal;
    param.N = size(X,1); % number of nodes in the graph
    param.J = param.N * param.S; % total number of atoms
    K = max(param.K);
    param.c = 1; % spectral control parameters
    param.mu = 1;%1e-2; % polynomial regularizer paremeter
    param.y = X; %signals
    param.y_size = size(param.y,2);
    param.T0 = 2; %sparsity level (# of atoms in each signals representation)
    my_max = zeros(1,param.S);
    
    %% initialise learned data
    
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
            % % %         % For the generation of control kernels coefficients
            % % %         originalKernels = param.alpha;
            % % %         filename = 'originalKernels.mat';
            % % %         save(filename,'originalKernels');
            
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
        %% optimise with regard to x
        disp(['Epoch... ',num2str(big_epoch)]);
        x = OMP_non_normalized_atoms(learned_dictionary,param.y, param.T0);
        
        switch starting_case
            case 1
                s = 0;
            case 2
                s = 1;
        end
        
        if mod(big_epoch + s,2) ~= 0
            %optimise with regard to W
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
        else
            % Set up the elements for the optimization problem
            %Optimize with regard to alpha
            [temp_alpha, my_max] = coefficient_update_interior_point(param.y,x,param,'sdpt3',big_epoch,my_max,flag);
            for j = 1:param.S
                param.alpha{j} = temp_alpha((j-1)*(K+1)+1:j*(K+1))';
            end
            [learned_dictionary, param] = construct_dict(param);
        end
        
        %% Plot the kernels
        
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
        
        if big_epoch == 6
            figure('Name','Last Test on kernel behavior')
            hold on
            for s = 1 : param.S
                plot(param.lambda_sym(2:length(param.lambda_sym)),g_ker(2:length(param.lambda_sym),s));
            end
            hold off
            
            filename = strcat(num2str(attempt_n),'a');
            fullfile = [path,'Results\',filename];
            saveas(gcf,fullfile,'png');
        end
        
    end
    
    figure('Name','Final kernels')
    hold on
    for s = 1 : param.S
        plot(param.lambda_sym(2:length(param.lambda_sym)),g_ker(2:length(param.lambda_sym),s));
    end
    hold off
    
    filename = strcat(num2str(attempt_n),'b');
    fullfile = [path,'Results\',filename];
    saveas(gcf,fullfile,'png');
    
    CoefMatrix_Pol = OMP_non_normalized_atoms(learned_dictionary,TestSignal, param.T0);
    errorTesting_Pol = sqrt(norm(TestSignal - learned_dictionary*CoefMatrix_Pol,'fro')^2/size(TestSignal,2));
    disp(['The total representation error of the testing signals is: ',num2str(errorTesting_Pol)]);
    sum_kernels = sum(param.kernel,2);
    
    % % % % Save the reconstructed signal
    % % % rebuiltSignal = learned_dictionary*CoefMatrix_Pol;
    % % % filename = 'rebuiltSignal.mat';
    % % % save(filename,'rebuiltSignal');
    
    %%
    %constructed graph needs to be tresholded, otherwise it's too dense
    %fix the number of desired edges here at nedges
    nedges = 4*29;
    final_Laplacian = treshold_by_edge_number(param.Laplacian, nedges);
    final_W = learned_W.*(final_Laplacian~=0);
    
    %% Save results to file
    alpha_coeff = zeros(K+1,2);
    for i = 1:param.S
        alpha_coeff(:,i) = param.alpha{i};
    end    
    %% Verifying the results with the precision recall function
    % Save the results in a file in which we have also the reconstruction error
    % such that we can compare it with the algo imposes the kernels' behavior
    % from the beginning
        learned_Laplacian = param.Laplacian;
        [optPrec, optRec, opt_Lapl] = precisionRecall(true_Laplacian, learned_Laplacian);
        filename = strcat(path,'Results\','ouput_PrecisionRecall',num2str(attempt_n),'.mat');
        save(filename,'opt_Lapl','optPrec','optRec','errorTesting_Pol');
    
    %% Verifying the l2-norm of kernels, X and dictionary
    % Kernels
        kernels_norm = norm(alpha_coeff - reference_kernels);
    % Sparsity matrix
%         X_norm = norm(CoefMatrix_Pol - reference_sparsity(:,901:1000));
    % Dictionary
        D_norm = norm(learned_dictionary - reference_D);
    % Weights
        W_norm = norm(final_W - reference_W);
        
    %% Save results to file
    filename = [path,'results\Output_results.mat'];
    save(filename,'final_Laplacian','final_W','alpha_coeff', 'g_ker','CoefMatrix_Pol','errorTesting_Pol','TrainSignal','TestSignal','sum_kernels','learned_dictionary');
    
    filename = [path,'results\l2_norms.mat'];
    save(filename,'kernels_norm','D_norm','W_norm');
end