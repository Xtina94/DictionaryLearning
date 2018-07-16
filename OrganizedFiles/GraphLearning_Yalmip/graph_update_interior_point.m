function [learned_dictionary, learned_w, learned_L_powers, cpuTime, param] = graph_update_interior_point(Data,CoefMatrix,param,sdpsolver,learned_W)

N = param.N;
M = param.M;
c = param.c;
epsilon = param.epsilon;
S = param.S;
K = max(param.K);

% for i = 1:param.S
%    alpha((K + 1)*(i-1) + 1:(K + 1)*i,1) = param.alpha{i}; 
% end

% Define the set of vertices connected to the vertex examined, basing the
% assumption on the W learned at the preceding operation
ops = size(Data,2);
Laplacian = sdpvar(ops,ops);
W = sdpvar(N,N);
for i = 1:size(Data,1)
    Ni = find(learned_W(i,:));
    for j = 1:length(Ni)
        Laplacian(i,:) = Laplacian(i,:) + W(i,Ni(j))*(Data(i,:)/sqrt(sum(W(i,:))) - Data(Ni(j),:)/sqrt(sum(W(Ni(j),:))));
    end
    Laplacian(i,:) = Laplacian(i,:)/sqrt(sum(W(i,:)));
end

% L = diag(sum(W,2)) - W; % combinatorial Laplacian
% Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian

% The laplacian powers
Laplacian_powers = cell(K+1,1);
for k = 0 : K
    Laplacian_powers{k + 1} = Laplacian^k;
end

% The eigenvalues and their powers

% % % [eigenMat, eigenVal] = eig(Laplacian);
% % % Lambda = sort(diag(eigenVal));
% % % Lambda_matrix(:,2) = Lambda;
% % % for i = 1:max(param.K) + 1
% % %     Lambda_matrix(:,i) = Lambda_matrix(:,2).^(i-1);
% % % end

%% Construct the objective function

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

% PhiPhiT = Phi*Phi';
% X = norm(Data,'fro')^2 - 2*YPhi*alpha + alpha'*(PhiPhiT + mu*eye(size(PhiPhiT,2)))*alpha;

% Localized atoms assumption:

L_power_mx = zeros((K+1)*size(Laplacian,1),size(Laplacian,2)); % A vector of laplacian matrices

for i = 1:K+1
    L_power_mx((i-1)*size(Laplacian,1) + 1:i*size(Laplacian,1),:) = Laplacian_powers{i};  
end

loc = 0;
for i = 1:N
    for j = 1:S*M
        loc = loc + norm(alpha'*L_power_mx,1);        
    end
end

% Final objective function:

X = norm(Data,'fro')^2 - 2*YPhi*alpha + loc;

%% Define the constraints 

BA = kron(eye(S),Lambda(1:size(Lambda,1),:));
BB = kron(ones(1,S),Lambda(1:size(Lambda,1),:));
la = length(BA*alpha);
lb = length(BB*alpha);

F = (BA*alpha <= c*ones(la,1))...
    + (BA*alpha >= 0.001*epsilon*ones(la,1))...
    + (BB*alpha <= (c+0.1*epsilon)*ones(lb,1))...
    + (BB*alpha >= ((c-0.1*epsilon)*ones(lb,1)));

%---------------------------------------------------------------------
% Solve the SDP using the YALMIP toolbox 
%---------------------------------------------------------------------

if strcmp(sdpsolver,'sedumi')
    diagnostics = optimize(F,X,sdpsettings('solver','sedumi','sedumi.eps',0,'sedumi.maxiter',200));
elseif strcmp(sdpsolver,'sdpt3')
    diagnostics = optimize(F,X,sdpsettings('solver','sdpt3'));
    elseif strcmp(sdpsolver,'mosek')
    diagnostics = optimize(F,X,sdpsettings('solver','mosek'));
    %sdpt3;
else
    error('??? unknown solver');
end

double(X);
param.objective(param.big_epoch) = double(X);
cpuTime = diagnostics.solveroutput.info.cputime;
learned_w = double(W);
learned_L = double(Laplacian);
learned_L_powers = cell(K+1,1);
for k = 0 : K
    learned_L_powers{k + 1} = learned_L^k;
end

for i=1:param.S
    learned_dict{i} = zeros(param.N);
end

for k = 1 : max(param.K)+1
    for i=1:param.S
        learned_dict{i} = learned_dict{i} + param.alpha{i}(k)*learned_L_powers{k};
    end
end

learned_dictionary = [learned_dict{1}];
for i = 2: param.S
    learned_dictionary = [learned_dictionary, learned_dict{i}];
end
