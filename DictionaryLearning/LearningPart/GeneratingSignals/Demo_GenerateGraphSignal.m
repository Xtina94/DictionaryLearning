% Generation of artificial data based on "How to learn a graph from smooth signal" 
% from Kalofolias. In this script I will generate several signals based on:
% 1. Random_geometric graph (RGG);
% 2. Non-uniform signal;
% In both the cases I smooth the signal through several techniques, like:
% 1. Tikhonov regularization;
% 2. Generative model;
% 3. Heat diffusion;

clear all;
close all;
load testdata.mat
path = 'C:\Users\Cristina\Documents\GitHub\DictionaryLearning\LearningPart\GeneratingSignals\DataSets\';

m = 100; %number of nodes
l = 1000; %signal length
sigma = 0.2; %std for Random Geometric Graph

%% Obtaining the corresponding weight and Laplacian matrices + the eigen decomposiotion parameters
% Sample uniformly from [0,1]
uniform_values = unifrnd(0,1,[1,m]);
k = 0.975;
[W] = random_geometric(sigma,m,uniform_values,k);
L = diag(sum(W,2)) - W;
Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2);
% Check for the Laplacian Norm
laplacian_norm = norm(Laplacian);
Lap_rank = rank(Laplacian);

[eigenVect, eigenVal] = eig(Laplacian);
[lambda_sym,index_sym] = sort(diag(eigenVal));

%% Obtaining the gaussian r.d. signal
causality_coeff = 1;
switch causality_coeff
    case 1
        d = m;
        cases = l;
% % %         sigma = rand(d); % generate a random n x n matrix
% % %         sigma = 0.5*(sigma+sigma');
% % %         sigma = sigma + d*eye(d);
        sigma = diag(rand(1,d));
        mu = zeros(1,d);
        R = mvnrnd(mu,sigma,cases);
        p = mvnpdf(R,mu,sigma);
        P = zeros(d,cases);
        for i = 1:d
            P(i,:) = p;
        end
        % % % figure('Name','Normal distribution signal')
        % % % surf(1:l,1:m,P);
        X_0 = R';
    case 2
        X_0 = 100*randn(m,l);
end

my_alpha = 10;
flag = 2;
switch flag
    case 1
        % Tikhonov regularization
        smoothing_method = 'Tikhonov';
        X_smooth = tikhonov(X_0,eigenVect,lambda_sym,my_alpha,m);
    case 2
        % Heat Kernel regularization
        smoothing_method = 'HeatKernel';
        X_smooth = heat(X_0,eigenVect,lambda_sym,my_alpha,m);
    case 3
        % no regularization
        smoothing_method = 'NoReg';
        X_smooth = X_0;
end

n_vertices = sum(W(:)>0);

figure('Name','Weight matrix')
surf(W)

figName = strcat(path,'WeightMatrix_',smoothing_method,'.jpg');
% % % saveas(gcf,figName);

figure('Name','Smoothed signal')
surf(X_smooth)

figName = strcat(path,'SmoothedSignal_',smoothing_method,'.jpg');
% % % saveas(gcf,figName);

filename = strcat(path,'SyntheticData_Gaussian',smoothing_method,'.mat');
save(filename,'X_smooth','W','Laplacian','eigenVal','eigenVect','lambda_sym','A','XCoords','YCoords','n_vertices');
