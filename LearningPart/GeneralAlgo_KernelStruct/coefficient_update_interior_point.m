function alpha = coefficient_update_interior_point(Data,CoefMatrix,param,sdpsolver)

N = param.N;
c = param.c;
epsilon = param.epsilon;
mu = param.mu;
S = param.S;
q = sum(param.K)+S;

alpha = polynomial_construct_low(param);
K = max(param.K);
Laplacian_powers = param.Laplacian_powers;
Lambda = param.lambda_power_matrix;

% Uber dataset
% % % param.thresh = param.percentage+21;

% Sythetic dataset
param.thresh = param.percentage+19;

thresh = param.thresh;

B1 = sparse(kron(eye(S),Lambda(1:size(Lambda,1),:)));
B2 = kron(ones(1,S),Lambda(1:size(Lambda,1)- thresh,:));
B3 = sparse(kron(eye(S),Lambda(size(Lambda,1)-param.percentage+1:size(Lambda,1),:)));

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

%-----------------------------------------------
% Define the Objective Function
%-----------------------------------------------

X = norm(Data,'fro')^2 - 2*YPhi*alpha + alpha'*(PhiPhiT + mu*eye(size(PhiPhiT,2)))*alpha;

%-----------------------------------------------
% Define Constraints
%----------------------------------------------- 

F = (B1*alpha <= c*ones(l1,1))...
    + (-B1*alpha <= -0*epsilon*ones(l1,1))...
    + (B2*alpha <= (c+0.1*epsilon)*ones(l2,1))... 
    + (-B2*alpha <= -0.1*epsilon*ones(l2,1));

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