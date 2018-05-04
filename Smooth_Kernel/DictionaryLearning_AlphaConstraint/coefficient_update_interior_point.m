
function [alpha, objective] = coefficient_update_interior_point(Data,CoefMatrix,param,sdpsolver)

% =========================================================================
   %%  Update the polynomial coefficients using interior point methods
% =========================================================================
% Description: It learns the polynomial coefficients using interior points methods. 
% We use the sdpt3 solver in the YALMIP optimization toolbox to solve the
% quadratic program. Both are publicly available in the following links
% sdpt3: http://www.math.nus.edu.sg/~mattohkc/sdpt3.html
% YALMIP: http://users.isy.liu.se/johanl/yalmip/

%% Set parameterss

N = param.N;
c = param.c;
epsilon = 10*param.epsilon;
mu = param.mu;
S = param.S;
q = sum(param.K)+S;
K = max(param.K);
Laplacian_powers = param.Laplacian_powers;
Lambda = param.lambda_power_matrix;
alpha = param.alpha_vector;
thresh = param.percentage+5;
B1 = sparse(kron(eye(S),Lambda));
B2 = kron(ones(1,S),Lambda);
B2 = B2(1:param.N-thresh,:);
B3 = sparse(kron(eye(S),Lambda(1:param.N-thresh,:)));

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
l3 = length(B3*alpha);

%% Objective function

X = norm(Data,'fro')^2 - 2*YPhi*alpha + alpha'*(PhiPhiT + mu*eye(size(PhiPhiT,2)))*alpha;

%% Contraints

F = (B1*alpha <= c*ones(l1,1))...
    + (-B3*alpha <= -0.01*epsilon*ones(l3,1))...
    + (B2*alpha <= (c+(100*epsilon))*ones(l2,1))...
    + (-B2*alpha <= -(c-(100*epsilon))*ones(l2,1));


% % % F = [(B1*alpha <= 1*ones(l1,1)), (-B1*alpha <= 0*ones(l1,1)), (B2*alpha <= (c+epsilon)*ones(l2,1)), (-B2*alpha <= -(c-epsilon)*ones(l2,1))];

%% Solving the SDP

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

objective = double(X);

alpha=double(alpha);