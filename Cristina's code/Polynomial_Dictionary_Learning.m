function [recovered_Dictionary, output] = Polynomial_Dictionary_Learning(Y, param, rts)

% =========================================================================
%                  Polynomial Dictionary Learning Algorithm
% =========================================================================
% Description: The function implements the polynomial dictionary learning 
%               algorithm presented in: 
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
%      param.Laplacian_powers:      a cell that contains the param.K powers of the graph Laplacian 
%      param.lambda_sym:           eigenvalues of the normalized Laplacian
%      param.lambda_powers:        a cell that contains the param.K powers of the eigenvalues of the graph Laplacian 
%      param.numIteration:        (optional) maximum number of iterations. When it is not
%                                 provided, it is set to 100 
%      param.InitializationMethod: (optional) A method for intializing the dictionary. 
%                                  If
%                                  param.InitializationMethod ='Random_kernels'(default),initialize using the function initialize_dictionary(),
%                                  else if 
%                                  param.InitializationMethod ='GivenMatrix'use a initial given dictionary (e.g., Spectral Graph Wavelet Dictionary)                                 
%      param.quadratic:           (optional) if 0 (by default), it uses interior point methods to
%                                 solve the dictionary update step, otherwise uses ADMM.
%      param.plot_kernels:        (optional) if 0 (by default), it plots the learned kernel after each iteration   
%      param.displayProgress:     (optional) if 0 (by default), it prints the total mean square representation error.
%      
%                                  
%

%% Output:
%     recoveredDictionary: The recovered polynomial dictionary
%     output: structure that includes all the following fields
%     output.alpha: learned polynomial coefficients
%     output.CoefMatrix: Sparse codes of the training signals
%     output.ker: The value of the kernels in the eigenvalues

% =========================================================================

%% Set parameters
lambda_sym = param.lambda_sym;
%lambda_powers = param.lambda_powers;
Laplacian_powers = param.Laplacian_powers;

%% Length adaptation for the lambda power vector

percentage = 0;
K = param.K;
lambda_powers_gamma = param.lambda_powers;
lambda_powers_beta = param.lambda_powers;
% Lambda_cri = param.lambda_power_matrix(:,1:K - percentage +1);

for i = 1:param.N
    lambda_powers_gamma{1,i} = param.lambda_powers{1,i}(1:K - percentage +1);
    lambda_powers_beta{1,i} = param.lambda_powers{1,i}(1:percentage +1);
end

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
    %param.InitializationMethod = 'GivenMatrix';
end

color_matrix = ['b', 'r', 'g', 'c', 'm', 'k', 'y'];
 
%% Initializing the dictionary

if (strcmp(param.InitializationMethod,'Random_kernels')) 
    [Dictionary(:,1 : param.J)] = initialize_dictionary(param);
   
    
elseif (strcmp(param.InitializationMethod,'GivenMatrix'))
        Dictionary(:,1 : param.J) = param.initialDictionary(:,1 : param.J);  %initialize with a given initialization dictionary
else 
    disp('Initialization method is not valid')
end

%% --------- Graph Dictionary Learning Algorithm --------- %%

for iterNum = 1 : param.numIteration
    
%%Sparse Coding Step (OMP)
    CoefMatrix = OMP_non_normalized_atoms(Dictionary,Y, param.T0);
    
%%Dictionary Update Step          
    if (param.quadratic == 0)
        if (iterNum == 1)
            disp('solving the quadratic problem with YALMIP...')
        end
        [alpha, pseudo_ker, beta_vector] = coefficient_update_interior_point_cri(Y,CoefMatrix,param,percentage,'sdpt3');
    else
        if (iterNum == 1)
            disp('solving the quadratic problem with ADMM...')
        end
        [Q1,Q2, B, h] = compute_ADMM_entries(Y, param, Laplacian_powers, CoefMatrix);
        alpha = coefficient_upadate_ADMM(Q1, Q2, B, h);
    end
      
    if (param.plot_kernels == 1)
        g_ker = zeros(param.N, param.S);
        r = 0;
        for i = 1 : param.S
            for n = 1 : param.N
                p = 0;
                for l = 0 : param.K(i) - percentage
                    p = p +  alpha(l + 1 + r)*lambda_powers_gamma{n}(l + 1);
                end
                g_ker(n,i) = p;
            end
            %r = sum(param.K(1:i)) + i;
            r = sum(param.K(1 : i) - percentage) + i;
        end
    end
        
%% --------------------------Final kernels---------------------------------
% --------Unifying the gamma coefficients and the beta coefficients--------

    final_ker = zeros(param.N,param.S);

    g_ker_beta = zeros(param.N, param.S);
    for i = 1 : param.S
        for n = 1 : param.N
            p = 0;
            for l = 0 : percentage
                p = p +  beta_vector(l + 1)*lambda_powers_beta{n}(l + 1);
            end
            g_ker_beta(n,i) = p;
        end
    end

    for s = 1:param.S
        for n = 1:param.N
            final_ker(n,s) = g_ker(n,s)*g_ker_beta(n,s);
        end
    end
    
%% Dictionary update

    r = 0;
    for j = 1 : param.S
        D = zeros(param.N);
        for ii = 0 : param.K(j) - percentage
            D = D +  alpha(ii + 1 + r) * Laplacian_powers{ii + 1};
        end
        r = sum(param.K(1:j) - percentage) + j;
        Dictionary(:,1 + (j - 1) * param.N : j * param.N) = D;
    end

    if (iterNum>1 && param.displayProgress)
        output.totalError(iterNum - 1) = sqrt(sum(sum((Y-Dictionary * CoefMatrix).^2))/numel(Y));
        disp(['Iteration   ',num2str(iterNum),'   Total error is: ',num2str(output.totalError(iterNum-1))]);
    end
    
end

output.ker = g_ker;

%% Plot the progress

figure('Name','Final kernel')
    hold on
    for s = 1 : param.S
        plot(lambda_sym,final_ker(:,s),num2str(color_matrix(s)));
    end
    hold off

figure('Name','Alpha Kernel')
    hold on
    for s = 1 : param.S
        plot(lambda_sym,g_ker(:,s),num2str(color_matrix(s)));
    end
    hold off

figure('Name', 'Pseudo kernel')
    hold on
    plot(lambda_sym,pseudo_ker);
    hold off

figure('Name', 'Alpha')
    plot(alpha);

output.pseudo_ker = pseudo_ker;

output.CoefMatrix = CoefMatrix;
output.alpha =  alpha;
output.beta = beta_vector;
recovered_Dictionary = Dictionary;


