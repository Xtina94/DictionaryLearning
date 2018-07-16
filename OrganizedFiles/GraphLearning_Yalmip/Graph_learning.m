function [learned_dictionary, output] = Graph_learning(Y,param)

%% Initialize the weight matrix

uniform_values = unifrnd(0,1,[1,param.N]);
sigma = 0.2;
[learned_W,param.Laplacian] = random_geometric(sigma,param.N,uniform_values,0.6);
 
% [param.eigenMat, param.eigenVal] = eig(param.Laplacian);
% [param.lambdaSym,indexSym] = sort(diag(param.eigenVal));
% param.lambda_power_matrix(:,2) = param.lambdaSym;
% for i = 1:max(param.K) + 1
%     param.lambda_power_matrix(:,i) = param.lambda_power_matrix(:,2).^(i-1);
% end

%% Initialize D:
[initial_dictionary, param] = construct_dict(param);
Dictionary = initial_dictionary;

%% Graph learning algorithm

cpuTime = zeros(1,param.numIteration);
% g_ker = zeros(param.N, param.S);
        
for iterNum = 1 : param.numIteration
    
    param.big_epoch = iterNum;
    
    % X update step 
    CoefMatrix = OMP_non_normalized_atoms(Dictionary,Y, param.T0);
    
    % W update step   
    if (param.quadratic == 0)
        if (iterNum == 1)
            disp('solving the quadratic problem with YALMIP...')
        end
        [Dictionary, learned_W, learned_L_powers, cpuTm, param] = graph_update_interior_point(Y,CoefMatrix,param,'sdpt3',learned_W);
        cpuTime(iterNum) = cpuTm;
    else
        if (iterNum == 1)
            disp('solving the quadratic problem with ADMM...')
        end
        [Q1,Q2, B, h] = compute_ADMM_entries(Y, param, Laplacian_powers, CoefMatrix);
        alpha = coefficient_upadate_ADMM(Q1, Q2, B, h);
    end    
    
    % D update step

    r = 0;
    for j = 1 : param.S
        D = zeros(param.N);
        for ii = 0 : param.K(j)
            D = D +  alpha(ii + 1 + r) * learned_L_powers{ii + 1};
        end
        r = sum(param.K(1:j)) + j;
        Dictionary(:,1 + (j - 1) * param.N : j * param.N) = D;
    end
    
    param.L = diag(sum(learned_W,2)) - learned_W; % combinatorial Laplacian
    param.Laplacian = (diag(sum(learned_W,2)))^(-1/2)*param.L*(diag(sum(learned_W,2)))^(-1/2); % normalized Laplacian
    
    [Dictionary, param] = construct_dict(param);
    
    if (iterNum>1 && param.displayProgress)
        output.totalError(iterNum - 1) = sqrt(sum(sum((Y-Dictionary * CoefMatrix).^2))/numel(Y));
        disp(['Iteration   ',num2str(iterNum),'   Total error is: ',num2str(output.totalError(iterNum-1))]);
    end
end

output.cpuTime = cpuTime;
output.CoefMatrix = CoefMatrix;
output.W =  learned_W;
learned_dictionary = Dictionary;
end
