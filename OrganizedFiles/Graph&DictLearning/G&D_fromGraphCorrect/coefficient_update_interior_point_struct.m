function [param, cpuTime] = coefficient_update_interior_point_struct(Data,CoefMatrix,param,sdpsolver,roots)

% Set parameters
my_max = param.max;
N = param.N;
c = param.c;
epsilon = 10*param.epsilon;
mu = param.mu;
K = max(param.K);
S = param.S;
Laplacian_powers = param.Laplacian_powers;
Lambda = param.lambda_power_matrix;
thresh = param.thresh;
alpha = sdpvar((K+1)*param.S,1);

BA = sparse(kron(eye(S),Lambda(1:size(Lambda,1),:)));
BB = kron(ones(1,S),Lambda(1:size(Lambda,1),:));

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

la = length(BA*alpha);
lb = length(BB*alpha);

%-----------------------------------------------
% Define the Objective Function
%-----------------------------------------------

X = norm(Data,'fro')^2 - 2*YPhi*alpha + alpha'*(PhiPhiT + mu*eye(size(PhiPhiT,2)))*alpha;

%-----------------------------------------------
% Define the new constraints deriving by the kernels behavior
% Start supposing a smooth kernel
%-----------------------------------------------
if param.iterN < 0
    F = (BA*alpha <= c*ones(la,1))...
        + (-BA*alpha <= -0.001*epsilon*ones(la,1))...
        + (BB*alpha <= (c+0.1*epsilon)*ones(lb,1))...
        + (-BB*alpha <= ((c-0.1*epsilon)*ones(lb,1)));
%         + (alpha(1:8) >= 0.01);
else
    h = eye(param.S);
    h2 = ones(1,param.S);
    
    for j = 1:param.S
        if j ~= 1
            h(j,j) = 0;
            h2(j) = 0;
        end
    end
    
    % I suppose to start with a low frequency kernel
    B1 = kron(h,Lambda(1:size(Lambda,1) - thresh,:));
    B2 = kron(h2,Lambda(1:size(Lambda,1) - thresh,:));
    
% % %     B1 = kron(h,Lambda(1:thresh,:));
% % %     B2 = kron(h2,Lambda(1:thresh,:));
    
    for i = 2:param.S
        h = eye(param.S);
        h2 = ones(1,param.S);
        for j = 1:param.S
            if j ~= i
                h(j,j) = 0;
                h2(j) = 0;
            end
        end
        if mod(i,2) ~= 0 % We are facing a low frequency kernel
            B1 = B1 + kron(h,Lambda(1:size(Lambda,1) - thresh,:));
            B2 = B2 + kron(h2,Lambda(1:size(Lambda,1)- thresh,:));
% % %             B1 = B1 + kron(h,Lambda(1:thresh,:));
% % %             B2 = B2 + kron(h2,Lambda(1:thresh,:));
        else             % Otherwise we're having a high frequency kernel
            B1 = B1 + kron(h,Lambda(thresh + 1:size(Lambda,1),:));
            B2 = B2 + kron(h2,Lambda(thresh + 1:size(Lambda,1),:));
% % %             B1 = B1 + kron(h,Lambda(size(Lambda,1) - thresh + 1: size(Lambda,1),:));
% % %             B2 = B2 + kron(h2,Lambda(size(Lambda,1) - thresh + 1: size(Lambda,1),:));
        end
    end
    
    % Find the coefficients vector structure
    alpha_low = polynomial_construct_low(param,roots);
    alpha_high = polynomial_construct_high(param,roots);
    
    alpha = [alpha_low; alpha_high];
    
    l1 = length(B1*alpha);
    l2 = length(B2*alpha);
    
    %% The official constraints
    
    F = (B1*alpha >= 0.001*epsilon*ones(l1,1))...
        + (B2*alpha <= (c+1*epsilon)*ones(l2,1))...
        + (-B2*alpha <= (c-1*epsilon)*ones(l2,1));
end
%% Solving the SDP

if strcmp(sdpsolver,'sedumi')
    diagnostics = solvesdp(F,X,sdpsettings('solver','sedumi','sedumi.eps',0,'sedumi.maxiter',200));
elseif strcmp(sdpsolver,'sdpt3')
    diagnostics = solvesdp(F,X,sdpsettings('solver','sdpt3'));
elseif strcmp(sdpsolver,'mosek')
    diagnostics = solvesdp(F,X,sdpsettings('solver','mosek'));
    %sdpt3;
else
    error('??? unknown solver');
end

double(X);
param.objective(param.iterN) = double(X);
alpha = double(alpha);
cpuTime = diagnostics.solveroutput.info.cputime;
alpha = double(alpha);

for i = 1:param.S
    param.alpha{i} = alpha((param.K(i) + 1)*(i-1) + 1:(param.K(i) + 1)*i);
end
end