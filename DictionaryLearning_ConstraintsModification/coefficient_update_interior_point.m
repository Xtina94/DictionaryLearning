
function [alpha_coefficients, err] = coefficient_update_interior_point(Data,CoefMatrix,param,sdpsolver)

% =========================================================================
   %%  Update the polynomial coefficients using interior point methods
% =========================================================================
% Description: It learns the polynomial coefficients using interior points methods. 
% We use the sdpt3 solver in the YALMIP optimization toolbox to solve the
% quadratic program. Both are publicly available in the following links
% sdpt3: http://www.math.nus.edu.sg/~mattohkc/sdpt3.html
% YALMIP: http://users.isy.liu.se/johanl/yalmip/
% =========================================================================

    %% Set parameters

    N = param.N;
    c = param.c;
    epsilon = param.epsilon;
    mu = param.mu;
    S = param.S;
    m = param.percentage;
    beta_coefficients = param.beta_coefficients;
    Laplacian_powers = param.Laplacian_powers;
    Lambda = param.lambda_power_matrix;

    q = sum(param.K)+S;
    alpha_coefficients = sdpvar(q,1);
    K = max(param.K);
    lambda_powers = param.lambda_powers;
    
    
    %% Defining the column vectors Pnm (for gamma vector)

    B1 = sparse(kron(eye(S),Lambda));
    B2 = kron(ones(1,S),Lambda);


    Phi = zeros(S*K,1);
    
    for i = 1 : N
        r = 0;
        for s = 1 : S
            for k = 0 : K
                Phi(k + 1 + r,(i - 1)*size(Data,2) + 1 : i*size(Data,2)) = Laplacian_powers{k+1}(i,:)*CoefMatrix((s - 1)*N+1 : s*N,1 : end);
            end
            r = sum(param.K(1 : s)) + s;
        end
    end
    YPhi = (Phi*(reshape(Data',1,[]))')';
    PhiPhiT = Phi*Phi';

    l1 = length(B1*alpha_coefficients);
    %l2 = length(B2*gamma_coefficients);

    %% Defining the objective function (for gamma vector)
    % I minimize with respect to alpha_coefficients so the eigneVal now stay the same

    X = norm(Data,'fro')^2 - 2*YPhi*alpha_coefficients + alpha_coefficients'*(PhiPhiT + mu*eye(size(PhiPhiT,2)))*alpha_coefficients;

    %% Defining the constraints (for gamma vector)

    lambda_mx = zeros(N,m+1);
    for i = N-m:N
        tmp_lambda = lambda_powers{i};
        for j =1:K+1
            lambda_mx(i,j) = tmp_lambda(j);
        end
    end
    
%     F = [B1*alpha_coefficients <= c*ones(l1,1), -B1*alpha_coefficients <= 0*ones(l1,1), lambda_mx*alpha_coefficients(1:K+1) = zeros(N,1)]; 
    
    %% Setting the alpha_coefficients vector
    %==========================================================================
    % alpha_coefficients vector can be represented as:
    % ---|---|---| ... |---| |---| ... |---| |---| ... |---| -|
    % ___|___|___| ... |___| |___| ... |___| |___| ... |___|  |
    %  1   2   3       K-M+1 K-M+2      M+1   M+2        K    | * S kernels
    % |----- Block A ------| |---Block B---| |---Block C---| _|
    % NOTE: For now I suppose percentage >= K/2
    %==========================================================================
% % %     for s = 0:S-1
% % % 
% % %         %For block A
% % %         for i = 1:K - m + 1
% % %             for j = 0:i-1
% % %                 alpha_coefficients(i+s*21,1) = alpha_coefficients(i+s*21,1) + gamma(i-j)*beta_coefficients(j+1);
% % %             end
% % %         end
% % % 
% % %         %For block B
% % %         index = K - m + 1;
% % % 
% % %         for j = 1:2 * m - K
% % %             for z = 0:K - m
% % %                 alpha_coefficients(index+j+s*21,1) = alpha_coefficients(index+j+s*21,1) + gamma(index-z)*beta_coefficients(j+z);
% % %             end
% % %         end
% % % 
% % %         %For block C
% % %         for i = 2:K - m +1
% % %             for j = 2:K - m +1
% % %                 alpha_coefficients(m+i+s*21,1) = alpha_coefficients(m+i+s*21,1) + gamma(j)*beta_coefficients(m);
% % %             end
% % %         end
% % %     end
% % %     %alpha_coefficients = fliplr(alpha_coefficients);

% % %     %% Defining the column vectors Pnm
% % % 
% % %     B1 = sparse(kron(eye(S),Lambda));
% % %     B2 = kron(ones(1,S),Lambda);
% % % 
% % % 
% % %     Phi = zeros(S*(K+1),1);
% % %     for i = 1 : N
% % %         r = 0;
% % %         for s = 1 : S
% % %             for k = 0 : K
% % %                 Phi(k + 1 + r,(i - 1)*size(Data,2) + 1 : i*size(Data,2)) = Laplacian_powers{k+1}(i,:)*CoefMatrix((s - 1)*N+1 : s*N,1 : end);
% % %             end
% % %             r = sum(param.K(1 : s)) + s;
% % %         end
% % %     end
% % %     YPhi = (Phi*(reshape(Data',1,[]))')';
% % %     PhiPhiT = Phi*Phi';
% % % 
% % %     l1 = length(B1*alpha_coefficients);
% % %     l2 = length(B2*alpha_coefficients);
% % % 
% % %     %% Defining the objective function
% % %     % I minimize with respect to alpha_coefficients so the eigneVal now stay the same
% % % 
% % %     X = norm(Data,'fro')^2 - 2*YPhi*alpha_coefficients + alpha_coefficients'*(PhiPhiT + mu*eye(size(PhiPhiT,2)))*alpha_coefficients;
% % % 
% % %     %% Defining the constraints
% % % 
% % %     F = (B1*alpha_coefficients <= c*ones(l1,1)) + (-B1*alpha_coefficients <= 0*ones(l1,1));

    %%  Solve the SDP using the YALMIP toolbox

    if strcmp(sdpsolver,'sedumi')
        diagnostic = solvesdp(F,X,sdpsettings('solver','sedumi','sedumi.eps',0,'sedumi.maxiter',200));
    elseif strcmp(sdpsolver,'sdpt3')
        diagnostic = solvesdp(F,X,sdpsettings('solver','sdpt3'));
    elseif strcmp(sdpsolver,'mosek')
        diagnostic = solvesdp(F,X,sdpsettings('solver','mosek'));
        %sdpt3;
    else
        error('??? unknown solver');
    end

    double(X);
    err = diagnostic.problem;
    
    alpha_coefficients = double(alpha_coefficients);
    %error = yalmiperror(X);
end