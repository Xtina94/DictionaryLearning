%_________________________________________________________
% DEMO: generating kernels, signals and retreiving kernels
%_________________________________________________________

%% Generating Kernels

close all

load testdata.mat
% % % % load reference_dictionary.mat
% % % % load initial_sparsity_mx.mat

%% Set the parameters

param.N = 100; % number of nodes in the graph
param.S = 4;  % number of subdictionaries 
param.J = param.N * param.S; % total number of atoms 

number_sub = ones(1,param.S);
param.K = 20.*number_sub;
%param.initialDictionary = reference_dictionary;

param.T0 = 4; % sparsity level in the training phase
param.c = 1; % spectral control parameters
param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
param.mu = 1e-2; % polynomial regularizer paremeter

param.percentage = 15; % number of coefficients in the beta vector that I impose to know.
                 % NOTE: is not automatic that #lambdas as root of beta =
                 % percentage, this because while solving the beta
                 % polynomial I can hav invalid roots like complex values
                 % or values not belonging to the interval [0,2]
 
%% Compute the Laplacian and the normalized Laplacian operator 
    
L = diag(sum(W,2)) - W; % combinatorial Laplacian
param.Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian
[param.eigenMat, param.eigenVal] = eig(param.Laplacian); % eigendecomposition of the normalized Laplacian
[param.lambda_sym,index_sym] = sort(diag(param.eigenVal)); % sort the eigenvalues of the normalized Laplacian in descending order


%% Precompute the powers of the Laplacian

for k=0 : max(param.K)
    param.Laplacian_powers{k + 1} = param.Laplacian^k;
end
    
for s=1:param.N
    for i=0:max(param.K)
        param.lambda_powers{s}(i + 1) = param.lambda_sym(s)^(i);
        param.lambda_power_matrix(s,i + 1) = param.lambda_sym(s)^(i);
     end
end

%% Generate the kernels

output = generate_kernels(param, param.percentage);
alpha_coefficients = output.coefficients;

%% Generate the dictionary

Dictionary = zeros(param.N,param.S*param.N);
for s = 1 : param.S
    for i = 1 : param.K(s)+1
        Dictionary(:,((s-1)*param.N)+1:s*param.N) = Dictionary(:,((s-1)*param.N)+1:s*param.N) +  alpha_coefficients(s,i) .* param.Laplacian_powers{i};
    end
end

%% Generate the sparsity mx

TrainSignal = zeros(100,600);
X = sparsity_matrix_initialize(param,TrainSignal);

%% Obtain the signal

SampleSignal = Dictionary*X;


