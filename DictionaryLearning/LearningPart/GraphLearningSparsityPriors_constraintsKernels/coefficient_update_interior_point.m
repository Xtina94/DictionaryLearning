function [alpha, my_max] = coefficient_update_interior_point(Data,CoefMatrix,param,sdpsolver,big_epoch,my_max,flag)

switch flag 
    case 1
        % For sinthetic data 
        param.thresh = param.percentage + 65;
    case 2
        % For Uber data
        param.thresh = param.percentage + 10;
    case 3
        % For Tikhonov regularized data
        param.thresh = param.percentage + 65;
    case 4
        % For Heat regularized data
        param.thresh = param.percentage + 65;
    case 5
        % For Taylor approx kernel data
        param.thresh = param.percentage + 65;
end


N = param.N;
c = param.c;
epsilon = param.epsilon;
mu = param.mu;
S = param.S;
q = sum(param.K)+S;
alpha = sdpvar(q,1);
sub_alpha = sdpvar(q/param.S,2);

K = max(param.K);
Laplacian_powers = param.Laplacian_powers;
Lambda = param.lambda_power_matrix;
thresh = param.thresh;

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
% Define Constraints
%----------------------------------------------- 
% The reasoning is:
% If we are in the firs tot cycles of optimization then don't ask for the
% kernels to be also null in some lambdas; at the same time try to
% understand the behavior of these kernels. When a certain amount of
% iterations has passed then try to separate the kernels into high and low
% frequency kernels in order to impose their behavior towards 0
% with respect to their nature

% I decide that 2/3 is a good lambda position in the labda_sym vector for
% the approximation of the high frequency concept
high_freq_thr = round((length(param.lambda_sym))/2); 
% Find the minimum of the kernel functions
if big_epoch < 7 && big_epoch > 1
    for i = 1:param.S
        kernel_vect = param.kernel(:,i);
        my_max(i) = find(param.kernel == max(kernel_vect(2:length(kernel_vect))),1);
        if my_max(i) > length(param.kernel)
            my_max(i) = my_max(i) - length(param.kernel);
        end
    end
end

% ker_max = my_max;

if big_epoch < 7
    F = (BA*alpha <= c*ones(la,1))...
        + (-BA*alpha <= -0.001*epsilon*ones(la,1))...
        + (BB*alpha <= (c+0.1*epsilon)*ones(lb,1))...
        + (-BB*alpha <= ((c-0.1*epsilon)*ones(lb,1)));
else
    for i = 1:param.S
        sub_alpha(:,i) = alpha((i-1)*(param.K+1)+1:i*(param.K+1),1);
        if my_max(i)> high_freq_thr %it means that we're facing a high frequency kernel
            B3{i} = kron(eye(1),Lambda(1:param.percentage,:));
            B1{i} = kron(eye(1),Lambda(size(Lambda,1) - thresh + 1:size(Lambda,1),:));
            B2{i} = kron(ones(1),Lambda(size(Lambda,1)- thresh + 1:size(Lambda,1),:));
        else %otherwise we're having a low frequency kernel
            B3{i} = kron(eye(1),Lambda(size(Lambda,1)-param.percentage+1:size(Lambda,1),:));
            B1{i} = kron(eye(1),Lambda(1:thresh,:));
            B2{i} = kron(ones(1),Lambda(1:thresh,:));
        end
        l3(i) = length(B3{i}*sub_alpha(:,i));
        l1(i) = length(B1{i}*sub_alpha(:,i));
        l2(i) = length(B2{i}*sub_alpha(:,i));
    end
    
        myB2 = B2{1};
        for i = 2:param.S
            myB2 = [myB2 B2{i}];
        end
        B2 = myB2;
    
        F = (-B1{1}*sub_alpha(:,1) <= -0.001*epsilon*ones(l1(1),1))...
            + (B2*alpha <= (c+1*epsilon)*ones(l2(1),1))...
            + (-B2*alpha <= (c-1*epsilon)*ones(l2(1),1))...
            + (B3{1}*sub_alpha(:,1) <= 0.001*epsilon*ones(l3(1),1))...
            + (-B3{1}*sub_alpha(:,1) <= 0*ones(l3(1),1));
        
    for i = 2:param.S
        F = F + (-B1{i}*sub_alpha(:,i) <= -0.001*epsilon*ones(l1(i),1))...
            + (B2*alpha <= (c+1*epsilon)*ones(l2(i),1))...
            + (-B2*alpha <= (c-1*epsilon)*ones(l2(i),1))...
            + (B3{i}*sub_alpha(:,i) <= 0.001*epsilon*ones(l3(i),1))...
            + (-B3{i}*sub_alpha(:,i) <= 0*ones(l3(i),1));
    end
end

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