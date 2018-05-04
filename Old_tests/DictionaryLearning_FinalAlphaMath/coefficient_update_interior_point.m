
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

% % % %% Obtaining the beta coefficients
% % % m = param.percentage;
% % % [param.beta_coefficients, rts] = retrieve_betas(param);
% % % 
% % % % % % gammaCoeff = rand(K-m+1,1);
% % % % % % alpha = polynomial_construct(param,gammaCoeff);
% % % alpha = polynomial_construct(param);
% % % 
% % % %% Verify alpha goes to 0 for the roots of the polynomial
% % % 
% % % vand_eig = zeros(m,param.S*(K+1));
% % % for s = 0:param.S-1
% % %     for i = 1:K+1
% % %         for j = 1:m
% % %             vand_eig(j,i+s*(K+1)) = rts(j)^(i-1);
% % %         end
% % %     end
% % % end
% % % prova1 = double(vand_eig*alpha);

%% Set up the elements for the optimization problem

alpha = param.alpha;
% % % jolly = param.percentage;
jolly = 8;
B1 = sparse(kron(eye(S),Lambda));
B1 = B1(1:param.N-jolly,:);
B2 = kron(ones(1,S),Lambda);
B2 = B2(1:param.N-jolly,:);

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
F = [(B1*alpha <= c*ones(l1,1)), (-B1*alpha <= 0*ones(l1,1)), (B2*alpha <= (c+epsilon)*ones(l2,1)), (-B2*alpha <= -(c-epsilon)*ones(l2,1))];

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