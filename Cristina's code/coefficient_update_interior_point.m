
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
%--------------------------------------------------------------------------
%Cristina: let's try to define the object gamma as an sdpvar vector of
%dimension=dimension of the vector of the remaining coefficients after
%having imposed the constraint on the highest eigenvalues
%--------------------------------------------------------------------------
percentage = 10;
gamma = sdpvar(K - percentage+1,1);
[beta, rts] = pol_reduction(param.lambda_sym,percentage);
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

alpha = sdpvar(q,1); 
Laplacian_powers = param.Laplacian_powers;
Lambda = param.lambda_power_matrix;

%--------------------------------------------------------------------------
% Cristina: Setting the alpha vector
%==========================================================================
% alpha vector can be represented as:
% ---|---|---| ... |---| |---| ... |---| |---| ... |---| -|
% ___|___|___| ... |___| |___| ... |___| |___| ... |___|  |
%  1   2   3       K-M+1 K-M+2      M+1   M+2        K    | * S kernels
% |----- Block A ------| |---Block B---| |---Block C---| _|
% NOTE: For now I suppose percentage >= K/2
%==========================================================================
% % % for s = 0:S-1
% % %     
% % %     %For block A
% % %     for i = 1:K-percentage+1
% % %         for j = 0:i-1
% % %             alpha(i+s*21,1) = alpha(i+s*21,1) + gamma(i-j)*beta(j+1);
% % %         end
% % %     end
% % % 
% % %     %For block B
% % %     index = K-percentage+1;
% % % 
% % %     for j = 1:2*percentage-K
% % %         for z = 0:K-percentage
% % %             alpha(index+j+s*21,1) = alpha(index+j+s*21,1) + gamma(index-z)*beta(j+z);
% % %         end
% % %     end
% % % 
% % %     %For block C
% % %     for i = 2:K-percentage+1
% % %         for j = 2:K-percentage+1
% % %             alpha(percentage+i+s*21,1) = alpha(percentage+i+s*21,1) + gamma(j)*beta(percentage);
% % %         end
% % %     end
% % % end
% % % %alpha = fliplr(alpha);

%%-------------------------------------------------------------------------
% New math
%%-------------------------------------------------------------------------

for s = 0:S-1
   
    %For block A
    for index = 1:K-percentage+1
        for j = 1:index
            alpha(index+s*21,1) = alpha(index+s*21,1) + gamma(j)*beta(index-j+1);
        end
    end

    %For block B

    for index = K-percentage+2:percentage+1
        for j = 1:K-percentage+1
            alpha(index+s*21,1) = alpha(index+s*21,1) + gamma(j)*beta(index-j+1);
        end
    end

    %For block C
    index = percentage;
    for i = 2:K-percentage+1
        for j = 2:K-percentage+1
            alpha(index+i+s*21,1) = alpha(percentage+i+s*21,1) + gamma(j)*beta(percentage-j+i+1);
        end
    end  
end
alpha = fliplr(alpha);
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

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

% % F = set(B1*alpha <= c*ones(l1,1))...
% %     + set(-B1*alpha <= 0*ones(l1,1))...
% %     + set(B2*alpha <= (c+epsilon)*ones(l2,1))...
% %     + set(-B2*alpha <= -(c-epsilon)*ones(l2,1));
 
F = (B1*alpha <= c*ones(l1,1)) + (-B1*alpha <= 0*ones(l1,1));
%---------------------------------------------------------------------
% Cristina: adding a constraint to the lmi function F
%---------------------------------------------------------------------
% % % lambdas = param.lambda_sym;
% % % 
% % % new_alpha = polynomial_div(S,N,K,lambdas,percentage);
% % % F = addlmi(F,'');

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