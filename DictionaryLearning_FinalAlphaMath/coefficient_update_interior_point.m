
function alpha = coefficient_update_interior_point(Data,CoefMatrix,param,sdpsolver)

% =========================================================================
   %%  Update the polynomial coefficients using interior point methods
% =========================================================================
% Description: It learns the polynomial coefficients using interior points methods. 
% We use the sdpt3 solver in the YALMIP optimization toolbox to solve the
% quadratic program. Both are publicly available in the following links
% sdpt3: http://www.math.nus.edu.sg/~mattohkc/sdpt3.html
% YALMIP: http://users.isy.liu.se/johanl/yalmip/

% =========================================================================

%-----------------------------------------------
% Set parameters
%-----------------------------------------------
N = param.N;
c = param.c;
epsilon = param.epsilon;
mu = param.mu;
S = param.S;
q = sum(param.K)+S;
K = max(param.K);
Laplacian_powers = param.Laplacian_powers;
Lambda = param.lambda_power_matrix;

%% Obtaining the beta coefficients
m = param.percentage;
[param.beta_coefficients, rts] = retrieve_betas(param);

%% Write down the beta polynomial and the gamma polynomial

% % % syms betas(x);
% % % betas(x) = beta_coefficients(1)*x^(0);
% % % 
% % % for i = 2:length(beta_coefficients)
% % %     betas(x) = betas(x) + beta_coefficients(i)*x^(i-1); 
% % % end
% % % 
% % % %syms gammas(x);
% % % syms gamma0 gamma1 gamma2 gamma3 gamma4 gamma5;
% % % gammas(x) = gamma0*x^(0) + gamma1*x^(1) + gamma2*x^(2) + gamma3*x^(3) + gamma4*x^(4) + gamma5*x^(5);
% % % 
% % % alpha_polynomial = betas*gammas;
% % % alpha = alpha_polynomial;

% % % gammaCoeff = rand(K-m+1,1);
alpha = polynomial_construct(param);

%% Verify alpha goes to 0 for the roots of the polynomial

vand_eig = zeros(m,param.S*(K+1));
for s = 0:param.S-1
    for i = 1:K+1
        for j = 1:m
            vand_eig(j,i+s*(K+1)) = rts(j)^(i-1);
        end
    end
end
% % % prova1 = vand_eig*alpha;

%% Set up the elements for the optimization problem

B1 = sparse(kron(eye(S),Lambda));
B2 = kron(ones(1,S),Lambda);

Phi = zeros(S*(K+1),1);
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

l1 = length(B1*alpha);
l2 = length(B2*alpha);

%-----------------------------------------------
% Define the Objective Function
%-----------------------------------------------


X = norm(Data,'fro')^2 - 2*YPhi*alpha + alpha'*(PhiPhiT + mu*eye(size(PhiPhiT,2)))*alpha;

%-----------------------------------------------
% Define Constraints
%-----------------------------------------------
% % % F = (-B1*alpha <= 0*ones(l1,1));
F = (B1*alpha <= c*ones(l1,1)) + (-B1*alpha <= 0*ones(l1,1));

%---------------------------------------------------------------------
% Solve the SDP using the YALMIP toolbox 
%---------------------------------------------------------------------

if strcmp(sdpsolver,'sedumi')
    solvesdp(F,X,sdpsettings('solver','sedumi','sedumi.eps',0,'sedumi.maxiter',200))
elseif strcmp(sdpsolver,'sdpt3')
    solvesdp(F,X,sdpsettings('solver','sdpt3'));
    elseif strcmp(sdpsolver,'mosek')
    solvesdp(F,X,sdpsettings('solver','mosek'));
    %sdpt3;
else
    error('??? unknown solver');
end

double(X);

alpha=double(alpha);