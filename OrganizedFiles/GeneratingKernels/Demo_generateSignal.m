clear all
close all
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\GeneratingKernels\Results\');
path = 'C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\';

flag = 3;
switch flag
    case 1
        load '2LF_kernels.mat'
        alpha_1 = alpha_2;
        alpha_2 = alpha_5;
        kernels_type = 'LF';
        n_kernels = 2;
        k = 15; %polynomial degree
    case 2
        load '2HF_kernels.mat'
        load 'Output_results.mat'
        alpha_1 = alpha_coeff(:,1);
        alpha_2 = alpha_coeff(:,2);
        kernels_type = 'HF';
        n_kernels = 2;
        k = 15; %polynomial degree
    case 3
        load 'LF_heatKernel.mat'
        alpha_1 = LF_heatKernel;
        kernels_type = 'Heat';
        n_kernels = 1;
        k = 15; %polynomial degree
end

comp_alpha = zeros(k+1,n_kernels);
for i = 1:n_kernels
    comp_alpha(:,i) = eval(strcat('alpha_',num2str(i)));
end

m = 100;
l = 1000;

%% Obtaining the corresponding weight and Laplacian matrices + the eigen decomposition parameters

uniform_values = unifrnd(0,1,[1,m]);
sigma = 0.2;
[W] = random_geometric(sigma,m,uniform_values,0.6);
L = diag(sum(W,2)) - W;
Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2);

[eigenVect, comp_eigenVal] = eig(Laplacian);
[lambda_sym,index_sym] = sort(diag(comp_eigenVal));

%% Precompute the powers of the Laplacian

Laplacian_powers = cell(1,k+1);

for j = 0 : k
    Laplacian_powers{j + 1} = Laplacian^j;
end

%% Construct the dictionary

D = cell(1,n_kernels);
comp_D = zeros(m,n_kernels*m);
for i = 1:n_kernels
    D{i} = Laplacian_powers{1}*comp_alpha(1,i);
    for j = 2:k+1
        D{i} = D{i} + Laplacian_powers{j}*comp_alpha(j,i);
    end
    comp_D(:,(i-1)*m + 1:i*m) = D{i};
end

%% Generate the sparsity matrix

t0 = n_kernels;
X = Generate_sparsity(t0,m,n_kernels*m);

temp = comp_alpha(:,1);
for i = 2:n_kernels
    temp = [temp; comp_alpha(:,i)];
end
comp_alpha = temp;

%% Generate the signal through Y = DX
Y = comp_D*X;
TrainSignal = Y(:,1:80);
TestSignal = Y(:,81:100);
comp_X = X(:,81:100);
comp_train_X = X(:,1:80);
filename = strcat(path,'DataSet',num2str(kernels_type),'.mat');
save(filename,'TestSignal','TrainSignal','W');

comp_W = W;

%% Save the results to file

filename = strcat(path,'Comparison_datasets\Comparison',num2str(kernels_type),'.mat');
save(filename,'comp_alpha','comp_D','comp_X','comp_train_X','comp_W','comp_eigenVal');
