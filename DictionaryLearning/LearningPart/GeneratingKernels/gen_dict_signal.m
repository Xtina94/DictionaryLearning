clear all
close all

path = 'C:\Users\Cristina\Documents\GitHub\DictionaryLearning\LearningPart\GeneratingKernels\';
% load 'Original coefficients.mat'
load '2LF_kernels.mat'
load '2HF_kernels.mat'

alpha_1 = alpha_2;
alpha_2 = alpha_5;

load 'Output_results.mat'
alpha_1 = alpha_coeff(:,1);
alpha_2 = alpha_coeff(:,2);

n_kernels = 2;
k = 15; %polynomial degree
alpha = zeros(k+1,n_kernels);
for i = 1:n_kernels
    alpha(:,i) = eval(strcat('alpha_',num2str(i)));
end
m = 100;
l = 1000;

%% Obtaining the corresponding weight and Laplacian matrices + the eigen decomposiotion parameters

uniform_values = unifrnd(0,1,[1,m]);
sigma = 0.2;
[W] = random_geometric(sigma,m,uniform_values,0.6);
L = diag(sum(W,2)) - W;
Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2);
% Check for the Laplacian Norm
laplacian_norm = norm(Laplacian);
Lap_rank = rank(Laplacian);

[eigenVect, eigenVal] = eig(Laplacian);
[lambda_sym,index_sym] = sort(diag(eigenVal));

%% Precompute the powers of the Laplacian

Laplacian_powers = cell(1,k+1);

for j = 0 : k
    Laplacian_powers{j + 1} = Laplacian^j;
end

%% Construct the dictionary
D = cell(1,n_kernels);
my_D = zeros(m,n_kernels*m);
for i = 1:n_kernels
    D{i} = Laplacian_powers{1}*alpha(1,i);
    for j = 2:k+1
        D{i} = D{i} + Laplacian_powers{j}*alpha(j,i);
    end
    my_D(:,(i-1)*m + 1:i*m) = D{i};
end

%% Generate the sparsity matrix
t0 = n_kernels;
X = generate_sparsity(t0,m,l);

%% Generate the signal through Y = DX
Y = my_D*X;

filename = strcat(path,'DataSet_LF.mat');
save(filename,'Y','W');

filename = strcat(path,'Comparison_values.mat');
save(filename,'alpha','my_D','X','W');
