
function [alpha_low, alpha_high, objective_low, objective_high] = coefficient_update_interior_point(Data,CoefMatrix,param,sdpsolver)

% Set parameters

N = param.N;
c = param.c;
epsilon = 10*param.epsilon;
mu = param.mu;
S = param.S;
q = sum(param.K)+S;
K = max(param.K);
Laplacian_powers = param.Laplacian_powers;
Lambda = param.lambda_power_matrix;
thresh = param.percentage+5;

%% Low part
B1_low = sparse(kron(eye(S),Lambda));
B2_low = kron(ones(1,S),Lambda(1:param.N-thresh,:));
B3_low = sparse(kron(eye(S),Lambda(1:param.N-thresh,:)));

alpha_low = param.alpha_vector_low;

Phi_low = zeros(S*(K+1),1);
for i = 1 : N
         r = 0;
        for s_low = floor(S/2+1) : S
            for k = 0 : K
                Phi_low(k + 1 + r,(i - 1)*size(Data,2) + 1 : i*size(Data,2)) = Laplacian_powers{k+1}(i,:)*CoefMatrix((s_low - 1)*N+1 : s_low*N,1 : end);
            end
            r = sum(param.K(1 : s_low)) + s_low;
        end
end
YPhi_low = (Phi_low*(reshape(Data',1,[]))')';
PhiPhiT_low = Phi_low*Phi_low';

l1_low = length(B1_low*alpha_low);
l2_low = length(B2_low*alpha_low);
l3_low = length(B3_low*alpha_low);

%% Objective function

X_low = norm(Data,'fro')^2 - 2*YPhi_low*alpha_low + alpha_low'*(PhiPhiT_low + mu*eye(size(PhiPhiT_low,2)))*alpha_low;

%% Contraints

F_low = (B1_low*alpha_low <= c*ones(l1_low,1))...
    + (-B3_low*alpha_low <= -0.01*epsilon*ones(l3_low,1))...
    + (B2_low*alpha_low <= (c+(100*epsilon))*ones(l2_low,1))...
    + (-B2_low*alpha_low <= -(c-(100*epsilon))*ones(l2_low,1));

%% Solving the SDP

if strcmp(sdpsolver,'sedumi')
    solvesdp(F_low,X_low,sdpsettings('solver','sedumi','sedumi.eps',0,'sedumi.maxiter',200))
elseif strcmp(sdpsolver,'sdpt3')
    solvesdp(F_low,X_low,sdpsettings('solver','sdpt3'));
    elseif strcmp(sdpsolver,'mosek')
    solvesdp(F_low,X_low,sdpsettings('solver','mosek'));
    %sdpt3;
else
    error('??? unknown solver');
end

%% High part
B1_high = sparse(kron(eye(S),Lambda));
B2_high = kron(ones(1,S),Lambda(thresh+1:param.N,:));
B3_high = sparse(kron(eye(S),Lambda(thresh+1:param.N,:)));

alpha_high = param.alpha_vector_high;

Phi_high = zeros(S*(K+1),1);
for i = 1 : N
         r = 0;
        for s_high = 1 : floor(S/2)
            for k = 0 : K
                Phi_high(k + 1 + r,(i - 1)*size(Data,2) + 1 : i*size(Data,2)) = Laplacian_powers{k+1}(i,:)*CoefMatrix((s_high - 1)*N+1 : s_high*N,1 : end);
            end
            r = sum(param.K(1 : s_high)) + s_high;
        end
end
YPhi_high = (Phi_high*(reshape(Data',1,[]))')';
PhiPhiT_high = Phi_high*Phi_high';

l1_high = length(B1_high*alpha_high);
l2_high = length(B2_high*alpha_high);
l3_high = length(B3_high*alpha_high);

%% Objective function

X_high = norm(Data,'fro')^2 - 2*YPhi_high*alpha_high + alpha_high'*(PhiPhiT_high + mu*eye(size(PhiPhiT_high,2)))*alpha_high;

%% Contraints

F_high = (B1_high*alpha_high <= c*ones(l1_high,1))...
    + (-B3_high*alpha_high <= -0.01*epsilon*ones(l3_high,1))...
    + (B2_high*alpha_high <= (c+(100*epsilon))*ones(l2_high,1))...
    + (-B2_high*alpha_high <= -(c-(100*epsilon))*ones(l2_high,1));

%% Solving the SDP

if strcmp(sdpsolver,'sedumi')
    solvesdp(F_high,X_high,sdpsettings('solver','sedumi','sedumi.eps',0,'sedumi.maxiter',200))
elseif strcmp(sdpsolver,'sdpt3')
    solvesdp(F_high,X_high,sdpsettings('solver','sdpt3'));
    elseif strcmp(sdpsolver,'mosek')
    solvesdp(F_high,X_high,sdpsettings('solver','mosek'));
    %sdpt3;
else
    error('??? unknown solver');
end

objective_low = double(X_low);

alpha_low = double(alpha_low);

objective_high = double(X_high);

alpha_high = double(alpha_high);