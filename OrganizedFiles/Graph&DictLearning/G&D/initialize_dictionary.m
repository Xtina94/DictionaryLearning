function [Initial_Dictionary,param] = initialize_dictionary(param)

%======================================================
   %%  Dictionary Initialization
%======================================================


%% Input:
%         param.N:        number of nodes of the graph
%         param.J:        number of atoms in the dictionary 
%         param.S:        number of subdictionaries
%         param.eigenMat: eigenvectors of the graph Laplacian
%         param.c:        upper-bound on the spectral representation of the kernels 
%           
%% Output: 
%         Initial_Dictionary: A matrix for initializing the dictionary
%======================================================

[param.eigenMat, param.eigenVal] = eig(param.Laplacian);
param.lambda_sym = sort(diag(param.eigenVal));
J = param.J;
N = param.N;
S = param.S;
c = param.c;
K = max(param.K);
Initial_Dictionary = zeros(N,J);

for k=0 : max(param.K)
    param.Laplacian_powers{k + 1} = param.Laplacian^k;
end

param.lambda_powers = zeros(length(param.lambda_sym),K+1);
for k=0 : max(param.K)
    param.lambda_powers(:,k+1) = param.lambda_sym.^k;
end

for i = 1 : S
   
    tmpLambda = c * rand(param.N);
       
    if isempty(tmpLambda)
        disp('Initialization fails');
        exit;
    end
    
    tmpLambda = diag(tmpLambda(randperm(N)));
    Initial_Dictionary(:,1 + (i - 1) * N : i * N) = param.eigenMat * tmpLambda * param.eigenMat';
end
