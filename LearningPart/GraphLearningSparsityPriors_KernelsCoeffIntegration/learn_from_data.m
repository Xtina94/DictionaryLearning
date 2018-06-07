clear all
close all
addpath(genpath(pwd))

%% load data in variable X here (it should be a matrix #nodes x #signals)
sigma = 20;
threshold = 0.819;
m = 10;
l = 100;
[X,W,mn] = random_geometric(sigma, threshold, m, l);
learned_W = W;
L = diag(sum(W,2)) - W; % combinatorial Laplacian
param.Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian
% X = rand(10,100);
x_coord = rand([100,1]);
y_coord = rand([100,1]);

%% Smooth the signal
[param.eigenMat, param.eigenVal] = eig(param.Laplacian); % eigendecomposition of the normalized Laplacian
[param.lambda_sym,index_sym] = sort(diag(param.eigenVal)); % sort the eigenvalues of the normalized Laplacian in descending order

X_smooth = tykhonov(X,param.eigenMat,param.eigenVal,10,m);
X = X_smooth;

%% initailize parameters
param.N = size(X,1); % number of nodes in the graph
param.S = 2;  % number of subdictionaries 
param.J = param.N * param.S; % total number of atoms 
param.K = [15 15]; % polynomial degree of each subdictionary
K = max(param.K);
param.c = 1; % spectral control parameters
param.epsilon = 0.03;%0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
param.mu = 1;%1e-2; % polynomial regularizer paremeter
sigma = 20;
param.y = X(:,1:80); %signals
TestSignal = X(:,81:100);
param.y_size = size(param.y,2);
param.percentage = 5;

%% Load Dorina's data
load testdata.mat
param.y = TrainSignal;
param.N = size(param.y,1); % number of nodes in the graph
learned_W = W;
L = diag(sum(W,2)) - W; % combinatorial Laplacian
param.Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% generate dictionary polynomial coefficients from heat kernel
for i = 1:param.S
    param.t(i) = param.S - i + 1;
end

param.alpha = generate_coefficients(param);
disp(param.alpha); %cell array containing the two vectors (one for each heat kernel) made of the 
                   %16 coefficients for the representing polynomials


%% initialise learned data
param.T0 = 6; %sparsity level (# of atoms in each signals representation)
% % % [param.Laplacian, learned_W] = init_by_weight(param.N);
[learned_dictionary, param] = construct_dict(param);
alpha = 2; %gradient descent parameter, it decreases with epochs

for big_epoch = 1:8
    %% optimise with regard to x
    disp(['Epoch... ',num2str(big_epoch)]);
    x = OMP_non_normalized_atoms(learned_dictionary,param.y, param.T0);
    
    %% optimise with regard to W 
    if mod(big_epoch,2) < 0 %~= 0
        maxEpoch = 1; %number of graph updating steps before updating sparse codes (x) again
        beta = 10^(-2); %graph sparsity penalty
        old_L = param.Laplacian;
        [param.Laplacian, learned_W] = update_graph(x, alpha, beta, maxEpoch, param, learned_W);
        [learned_dictionary, param] = construct_dict(param);
        alpha = alpha*0.985; %gradient descent decreasing
    else
    %% optimize with regard of g(lambda)        
        [param.eigenMat, param.eigenVal] = eig(param.Laplacian); % eigendecomposition of the normalized Laplacian
        [param.lambda_sym,index_sym] = sort(diag(param.eigenVal)); % sort the eigenvalues of the normalized Laplacian in descending order
        param.lambda_power_matrix = zeros(param.N,K+1);
        for j = 0:K
            param.lambda_power_matrix(:,j+1) = param.lambda_sym.^(j);
        end
        
        param.alpha = update_coefficients(param.y,x,param,'sdpt3');
        alpha_coeff = cell(2,1);
        for i = 1:param.S
            alpha_coeff{i} = param.alpha((i-1)*(K+1)+1:i*(K+1));
        end
        param.alpha = alpha_coeff;
        [learned_dictionary, param] = construct_dict(param);
    end
end

%% Threshold the constructed graph
nedges = 4*29;
final_Laplacian = treshold_by_edge_number(param.Laplacian, nedges);
final_W = learned_W.*(final_Laplacian~=0);

%% eigenvalue decomposition
[eigVect, eigVal] = eig(final_Laplacian); 
[eigVal_sort, index_eigVal] = sort(diag(eigVal)); %Vector with the eigenvalues sorted in decreasing order

%% estimating powers of the eigenvalues
eigVal_powers = cell(1,16);
for j=1:param.N
    for i=0:max(param.K)
        eigVal_powers{j}(i + 1) = eigVal_sort(j)^(i);
     end
end

%% estimating the final kernels
g_ker = zeros(param.N, param.S);
r = 0;
for i = 1 : param.S
    for n = 1 : param.N
    p = 0;
    for l = 0 : param.K(i)
        p = p +  param.alpha{i}(l + 1)*eigVal_powers{n}(l + 1);
    end
    g_ker(n,i) = p;
    end
    r = sum(param.K(1:i)) + i;
end

%% plotting the graph
figure('Name','graph representation')   
gplot(X,[x_coord y_coord]);

%% plotting the kernels
figure('Name', 'Kernels')
hold on
for s = 1 : param.S
    plot(eigVal_sort,g_ker(:,s));
end
hold off
saveas(gcf,'Kernels','bmp');

filename = 'Results';
save(filename,'x','learned_dictionary','learned_W','g_ker');

CoefMatrix_Pol = OMP_non_normalized_atoms(learned_dictionary,TestSignal, param.T0);
errorTesting_Pol = sqrt(norm(TestSignal - learned_dictionary*CoefMatrix_Pol,'fro')^2/size(TestSignal,2));
disp(['The total representation error of the testing signals is: ',num2str(errorTesting_Pol)]);
