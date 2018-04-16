% ---------Kernel Retrieval code---------- %
clear all;
close all;

load output_Pol.mat
load testdata.mat

%% Setting the initial values

param.N = 100; % number of nodes in the graph
param.S = 4;  % number of subdictionaries 
param.J = param.N * param.S; % total number of atoms 

degree = 20;
percentage = length(output_Pol.beta) - 1;
number_sub = ones(1,param.S);
param.K = degree.*number_sub;
%param.initialDictionary = reference_dictionary;

K = max(param.K);
param.T0 = 4; % sparsity level in the training phase
param.c = 1; % spectral control parameters
param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
param.mu = 1e-2; % polynomial regularizer paremeter

%% Compute the Laplacian and the normalized Laplacian operator
    
L = diag(sum(W,2)) - W; % combinatorial Laplacian
param.Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian
[param.eigenMat, param.eigenVal] = eig(param.Laplacian); % eigendecomposition of the normalized Laplacian
[param.lambda_sym, index_sym] = sort(diag(param.eigenVal)); % sort the eigenvalues of the normalized Laplacian in descending order

%% Generating the laplacian powers

for k = 0 : K - percentage
    param.Laplacian_powers_alpha{k + 1} = param.Laplacian^k;
end

for k = 0 : percentage
    param.Laplacian_powers_beta{k + 1} = param.Laplacian^k;
end

%% Generating the lambda powers for the beta vector

for j = 1:param.N
    for i = 0:percentage
        param.lambda_powers_beta{j}(i + 1) = param.lambda_sym(j)^(i);
     end
end

%% Generating the lambda powers for the alpha vector

for j = 1:param.N
    for i = 0:K - (percentage)
        param.lambda_powers_alpha{j}(i + 1) = param.lambda_sym(j)^(i);
     end
end

%% Define the alpha vector coefficients

alpha_vector = zeros(param.S,K - (percentage - 1));
for i = 1 : param.S
    alpha_vector(i,:) = rand([1,K - (percentage - 1)]);
end

%% Define the beta vector coefficients

beta_vector = zeros(param.S,length(output_Pol.beta));
for i = 1 : param.S
    beta_vector(i,:) = output_Pol.beta';
end

%% kernel generation

kernel_alpha = zeros(param.S,param.N);
kernel_beta = zeros(param.S,param.N);
kernel = zeros(param.S,param.N);

D_alpha = cell(4,1);
D_beta = cell(4,1);
D = cell(4,1);

for i = 1:param.S
    for j = 1:param.N
        kernel_alpha(i,j) = alpha_vector(i,:)*(param.lambda_powers_alpha{j})';
        kernel_beta(i,j) = beta_vector(i,:)*(param.lambda_powers_beta{j})';
        kernel(i,j) = kernel_alpha(i,j)*kernel_beta(i,j);
    end
    
    for k = 0 : K - percentage
        D_alpha{i} =  alpha_vector(i,k+1)*param.Laplacian_powers_alpha{k + 1};
    end
    
    for m = 0 : percentage
        D_beta{i} =  beta_vector(i,m+1)*param.Laplacian_powers_beta{m + 1};
    end
    
    D{i} = D_alpha{i}*D_beta{i};
end

%% Generating sparsity matrix

X = zeros(param.S*size(D{1},1),size(TestSignal,2));
gauss_values = cell(1,size(X,2));
positions = cell(1,size(X,2));

for j = 1 : size(X,2)
    gauss_values{1,j} = randn(1,param.T0);
    positions{1,j} = randperm(size(X,1),param.T0);
    i = 1;
    while i <= param.T0
        X(positions{1,j}(1,i),j) = gauss_values{1,j}(1,i);
        i = i + 1;
    end
end

% NZ =zeros(4,size(X,2));

% for i = 1 : size(X,2)
%     NZ{i,1} = nonzeros(X(:,i));
%     sizes(i) = length(NZ{i,1});
% end

% colors = ['r', 'b', 'g', 'y', 'c'];

figure('Name', 'Imposed Kernels')
hold on
for i = 1:param.S
    plot(param.lambda_sym,kernel(i,:))
end
hold off




