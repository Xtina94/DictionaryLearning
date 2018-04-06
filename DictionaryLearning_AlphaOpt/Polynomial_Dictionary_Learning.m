function [recovered_Dictionary,output, err_or] = Polynomial_Dictionary_Learning(Y, param, initial_sparsity_mx)

    % =========================================================================
    %                  Polynomial Dictionary Learning Algorithm
    % =========================================================================
    % Description: The function implements the polynomial dictionary learning algorithm
    %               presented in:
    %               "Learning Parametric Dictionaries for Signals on Graphs",
    %                by D. Thanou, D. I Shuman, and P. Frossard
    %                arXiv: 1401.0887, January, 2014.
    %% Input:
    %      Y:                         Set of training signals
    %      param:                     Structure that includes all required parameters for the
    %                                 algorithm. It contains the following fields:
    %      param.N:                   number of nodes in the graph
    %      param.S:                   number of subdictionaries
    %      param.K:                   polynomial degree in each subdictionary
    %      param.T0:                  sparsity level in the learning phase
    %      param.Laplacian_powers:    a cell that contains the param.K powers of the graph Laplacian
    %      param.lambda_sym:          eigenvalues of the normalized Laplacian
    %      param.lambda_powers:       a cell that contains the param.K powers of the eigenvalues of the graph Laplacian
    %      param.numIteration:        (optional) maximum number of iterations. When it is not
    %                                 provided, it is set to 100
    %      param.InitializationMethod: (optional) A method for intializing the dictionary.
    %                                  If
    %                                  param.InitializationMethod ='Random_kernels'(default),initialize using the function initialize_dictionary(),
    %                                  else if
    %                                  param.InitializationMethod ='GivenMatrix'use a initial given dictionary (e.g., Spectral Graph Wavelet Dictionary)
    %      param.quadratic:           (optional) if 0 (by default), it uses interior point methods to
    %                                 solve the dictionary ipdate step, otherwise uses ADMM.
    %      param.plot_kernels:        (optional) if 0 (by default), it plots the learned kernel after each iteration
    %      param.displayProgress:     (optional) if 0 (by default), it prints the total mean square representation error.
    %
    %
    %

    %% Output:
    %     recoveredDictionary: The recovered polynomial dictionary
    %     output: structure that includes all the following fields
    %     output.alpha_coefficients: learned polynomial coefficients
    %     output.CoefMatrix: Sparse codes of the training signals

    % =========================================================================

    %% Set parameters

    lambda_sym = param.lambda_sym;
    lambda_powers = param.lambda_powers;
    Laplacian_powers = param.Laplacian_powers;

    if (~isfield(param,'displayProgress'))
        param.displayProgress = 0;
    end

    if (~isfield(param,'quadratic'))
        param.quadratic = 0;
    end

    if (~isfield(param,'plot_kernels'))
        param.plot_kernels = 0;
    end

    if (~isfield(param,'numIteration'))
        param.numIteration = 100;
    end

    if (~isfield(param,'InitializationMethod'))
        param.InitializationMethod = 'Random_kernels';
        % % %         param.InitializationMethod = 'GivenMatrix';
    end

    color_matrix = ['b', 'r', 'g', 'c', 'm', 'k', 'y'];

    %%  Graph Dictionary Learning Algorithm

    CoefMatrix = initial_sparsity_mx;

    % Estimation of the beta vector (fixed vector)
    m = param.percentage;
    
    g_ker_beta = zeros(param.N, param.S);
    for i = 1 : param.S
        for n = 1 : param.N
            p = 0;
            for l = 0 : m
                p = p +  param.beta_coefficients(l + 1)*lambda_powers{n}(l + 1);
            end
            g_ker_beta(n,i) = p;
        end
    end

    for iterNum = 1 : param.numIteration

        % Coefficients Update Step
        if (param.quadratic == 0)
            if (iterNum == 1)
                disp('solving the quadratic problem with YALMIP...')
            end
            [alpha_coefficients, err] = coefficient_update_interior_point(Y,CoefMatrix,param,'sdpt3');
% % %             [gamma_coefficients, err] = coefficient_update_interior_point(Y,CoefMatrix,param,'sdpt3');
        else
            if (iterNum == 1)
                disp('solving the quadratic problem with ADMM...')
            end
            [Q1,Q2, B, h] = compute_ADMM_entries(Y, param, Laplacian_powers, CoefMatrix);
             alpha_coefficients = coefficient_upadate_ADMM(Q1, Q2, B, h);
        end

        %%Generating the kernels
        if (param.plot_kernels == 1)
            g_ker = zeros(param.N, param.S);
            r = 0;
            for i = 1 : param.S
                for n = 1 : param.N
                    p = 0;
                    for l = 0 : param.K(i)
                        p = p +  alpha_coefficients(l + 1 + r)*lambda_powers{n}(l + 1);
% % %                         p = p +  gamma_coefficients(l + 1 + r)*lambda_powers{n}(l + 1);
                    end
                    g_ker(n,i) = p;
                end
                r = sum(param.K(1:i)) + i;
            end

            % % %             figure()
            % % %             hold on
            % % %             for s = 1 : param.S
            % % %                 plot(lambda_sym,g_ker(:,s),num2str(color_matrix(s)));
            % % %             end
            % % %             hold off
        end

        %% Dictionary update step

        r = 0;
        for j = 1 : param.S
            D = zeros(param.N);
            for ii = 0 : param.K(j)
                D = D +  alpha_coefficients(ii + 1 + r) * Laplacian_powers{ii + 1};
            end
            r = sum(param.K(1:j)) + j;
            Dictionary(:,1 + (j - 1) * param.N : j * param.N) = D;
        end

        %% Plot the progress

        if (iterNum>1 && param.displayProgress)
            output.totalError(iterNum - 1) = sqrt(sum(sum((Y-Dictionary * CoefMatrix).^2))/numel(Y));
            disp(['Iteration   ',num2str(iterNum),'   Total error is: ',num2str(output.totalError(iterNum-1))]);
        end

        % Sparse Coding Step (OMP)
        CoefMatrix = OMP_non_normalized_atoms(Dictionary,Y, param.T0);

    end
 
    figure()
    hold on
    for s = 1 : param.S
        plot(lambda_sym,g_ker(:,s),num2str(color_matrix(s)));
    end
    hold off

    output.CoefMatrix = CoefMatrix;
    output.alpha_coefficients =  alpha_coefficients;
% % %     output.gamma_coefficients =  gamma_coefficients;
    recovered_Dictionary = Dictionary;
    err_or = err;
end