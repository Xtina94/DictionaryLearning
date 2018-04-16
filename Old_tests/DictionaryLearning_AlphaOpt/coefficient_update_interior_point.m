
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
    mu = param.mu;
    S = param.S;
    K = max(param.K);
    m = param.percentage;
    q = sum(param.K)+S;

    Laplacian_powers = param.Laplacian_powers;
    Lambda = param.lambda_power_matrix;

    alpha_coefficients = sdpvar(q,1);
    gamma_coefficients = sdpvar(K - m + 1,1);

    %lambda_powers_beta = param.lambda_powers;
    %epsilon = param.epsilon;

    %% Retrieving the alpha_coefficients vector
    %==========================================================================
    % alpha_coefficients vector can be represented as:
    % ---|---|---| ... |---| |---| ... |---| |---| ... |---| -|
    % ___|___|___| ... |___| |___| ... |___| |___| ... |___|  |
    %  1   2   3       K-M+1 K-M+2      M+1   M+2        K    | * S kernels
    % |----- Block A ------| |---Block B---| |---Block C---| _|
    % NOTE: For now I suppose percentage >= K/2
    % =========================================================================

    for s = 0 : param.S - 1

        %For block A
        for i = 1 : K - m + 1
            for j = 1 : i
                alpha_coefficients(i+s*param.K(s + 1),1) = alpha_coefficients(i+s*param.K(s + 1),1) + gamma_coefficients(j)*param.beta_coefficients(i-j+1);
            end
        end

        %For block B
        for i = K - m + 2 : m + 1
            for j = 1 : K - m +1
                alpha_coefficients(i+s*param.K(s + 1),1) = alpha_coefficients(i+s*param.K(s + 1),1) + gamma_coefficients(j)*param.beta_coefficients(i-j+1);
            end
        end

        %For block C
        index = m;
        for i = 2 : K - m + 1
            for j = i : K - m + 1
                alpha_coefficients(index+i+s*param.K(s + 1),1) = alpha_coefficients(index+i+s*param.K(s + 1),1) + gamma_coefficients(j)*param.beta_coefficients(m-j+i+1);
            end
        end
    end

    %% Defining the column vectors Pnm (for gamma vector)

    B1 = sparse(kron(eye(S),Lambda));

    Phi = zeros(S*(K),1);

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

    %% Defining the objective function (for gamma vector)
    % I minimize with respect to alpha_coefficients so the eigneVal now stay the same

    X = norm(Data,'fro')^2 - 2*YPhi*alpha_coefficients + alpha_coefficients'*(PhiPhiT + mu*eye(size(PhiPhiT,2)))*alpha_coefficients;

    %% Defining the constraints (for gamma vector)

    alpha_coefficients = 100.*alpha_coefficients;
    F = (B1*alpha_coefficients <= c*ones(l1,1)) + (-B1*alpha_coefficients <= 100*0*ones(l1,1));


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
    err = diagnostic;

    alpha_coefficients = double(alpha_coefficients);
end