clear all
close all

%% Adding the paths
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\Optimizers'); %Folder conatining the yalmip tools
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\Comparison_datasets\'); %Folder containing the copmarison datasets
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\'); %Folder containing the training and verification dataset
path = 'C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DictionaryLearning\ThroughConstraints\Results\'; %Folder containing the results to save

%% Loaging the required dataset
flag = 1;
switch flag
    case 1
        load ComparisonDorina.mat
        load DataSetDorina.mat
    case 2
        load ComparisonLF.mat
        load DataSetLF.mat
    case 3
        load ComparisonUber.mat
        load DataSetUber.mat
end

%% Set the parameters

switch flag
    case 1 %Dorina
        param.S = 4;  % number of subdictionaries 
        param.epsilon = 0.05; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 20;
        param.N = 100; % number of nodes in the graph
        ds = 'Dataset used: Synthetic data from Dorina';
        ds_name = 'Dorina';
        param.percentage = 15;
        param.thresh = param.percentage+65;
    case 2 %Cristina
        param.S = 2;  % number of subdictionaries 
        param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
        param.N = 100; % number of nodes in the graph
        ds = 'Dataset used: data from Cristina';
        ds_name = 'Cristina'; 
        param.percentage = 8;
        param.thresh = param.percentage+60;
    case 3 %Uber
        param.S = 2;
        param.epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
        param.N = 29; % number of nodes in the graph
        ds = 'Dataset used: data from Uber';
        ds_name = 'Uber';
        param.percentage = 8;
        param.thresh = param.percentage+6;
end

param.J = param.N * param.S; % total number of atoms 
param.K = degree*ones(1,param.S);
param.T0 = 4; % sparsity level in the training phase
param.c = 1; % spectral control parameters
param.mu = 1e-2; % polynomial regularizer paremeter

%% Compute the Laplacian and the normalized laplacian operator
    
L = diag(sum(W,2)) - W; % combinatorial Laplacian
param.Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian
[param.eigenMat, param.eigenVal] = eig(param.Laplacian); % eigendecomposition of the normalized Laplacian
[param.lambda_sym,index_sym] = sort(diag(param.eigenVal)); % sort the eigenvalues of the normalized Laplacian in descending order

%% Compute the powers of the Laplacian

for k=0 : max(param.K)
    param.Laplacian_powers{k + 1} = param.Laplacian^k;
end
    
for j=1:param.N
    for i=0:max(param.K)
        param.lambda_powers{j}(i + 1) = param.lambda_sym(j)^(i);
        param.lambda_power_matrix(j,i + 1) = param.lambda_sym(j)^(i);
     end
end
    
%% Polynomial dictionary learning algorithm

param.InitializationMethod =  'Random_kernels';
param.displayProgress = 1;
param.numIteration = 3;
param.plot_kernels = 1; % plot thelearned polynomial kernels after each iteration
param.quadratic = 0; % solve the quadratic program using interior point methods

disp('Starting to train the dictionary');

[Dictionary_Pol,output_Pol]  = Polynomial_Dictionary_Learning(TrainSignal, param);

CoefMatrix_Pol = OMP_non_normalized_atoms(Dictionary_Pol,TestSignal, param.T0);
errorTesting_Pol = sqrt(norm(TestSignal - Dictionary_Pol*CoefMatrix_Pol,'fro')^2/size(TestSignal,2));
disp(['The total representation error of the testing signals is: ',num2str(errorTesting_Pol)]);

%% Compute the l-2 norms

lambda_norm = 'is 0 since here we are learning only the kernels'; %norm(comp_eigenVal - eigenVal);
alpha_norm = norm(comp_alpha - output_Pol.alpha);
X_norm = norm(comp_X - CoefMatrix_Pol);
D_norm = norm(comp_D - Dictionary_Pol);
W_norm = 'is 0 since here we are learning only the kernels';

%% Compute the average CPU_time

avgCPU = mean(output_Pol.cpuTime);

%% Save the results to file

% The norms
filename = [path,'Norms.mat'];
save(filename,'lambda_norm','alpha_norm','X_norm','D_norm','W_norm');

% The Output data
filename = [path,'Output.mat'];
learned_alpha = output_Pol.alpha;
save(filename,'ds','Dictionary_Pol','learned_alpha','CoefMatrix_Pol','errorTesting_Pol','avgCPU');

% The kernels plot
figure('Name','Final Kernels')
hold on
for s = 1 : param.S
    plot(param.lambda_sym,output_Pol.g_ker(:,s));
end
hold off

filename = [path,'FinalKernels_plot.png'];
saveas(gcf,filename);

% The CPU time plot
xq = 0:0.2:param.numIteration;
figure('Name','CPU time per iteration')
vq2 = interp1(1:param.numIteration,output_Pol.cpuTime,xq,'spline');
plot(1:param.numIteration,output_Pol.cpuTime,'o',xq,vq2,':.');
xlim([0 param.numIteration]);

filename = [path,'AvgCPUtime_plot.png'];
saveas(gcf,filename);

