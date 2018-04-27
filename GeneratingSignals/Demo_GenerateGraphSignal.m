% Generation of artificial data based on "How to learn a graph from smooth signal" 
% from Kalofolias. In this script I will generate several signals based on:
% 1. Random_geometric graph (RGG);
% 2. Non-uniform signal;
% In both the cases I smooth the signal through several techniques, like:
% 1. Tikhonov regularization;
% 2. Generative model;
% 3. Heta diffusion;

close all;

m = 100; %number of nodes
l = 1000; %signal length
sigma = 20; %std for Random Geometric Graph
threshold = 0.8133; %threshold for the weigths

%% Generating the starting signal

[X,W,mn] = random_geometric(sigma,threshold,m,l);

%% Generating the Laplacian and its decomposition

L = diag(sum(W,2)) - W;
Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); 
[eigenVect, eigenVal] = eig(Laplacian);
[lambda_sym,index_sym] = sort(diag(eigenVal));

%% Smoothing the signal

% Tykhonov regularization

X_smooth = tykhonov(X,eigenVect,lambda_sym,10);



