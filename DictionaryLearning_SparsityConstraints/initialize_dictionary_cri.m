function Initial_Dictionary = initialize_dictionary_cri(param)

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


J = param.J;
N = param.N;
S = param.S;
c = param.c;
chosen_lambda = (param.chosen_lambda);
Initial_Dictionary = zeros(N,J);

for i = 1 : S
   
    tmpLambda = [c*rand(N - length(chosen_lambda),1) chosen_lambda];
       
    if isempty(tmpLambda)
        disp('Initialization fails');
        exit;
    end
    
    tmpLambda = diag(tmpLambda(randperm(N)));
    Initial_Dictionary(:,1 + (i - 1) * N : i * N) = param.eigenMat * tmpLambda * param.eigenMat';
end
