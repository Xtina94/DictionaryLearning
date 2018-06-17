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

m = 100; %number of nodes
l = 1000; %signal length
sigma = 20; %std for Random Geometric Graph
% threshold = 0.819; %threshold for the weigths

%% Obtaining the corresponding weight and Laplacian matrices + the eigen decomposiotion parameters
uniform_values = rand(m,l);
k = 3/4;
[W,mn,variance] = random_geometric(sigma,m,uniform_values,k);
L = diag(sum(W,2)) - W;
Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2);
Lap_rank = rank(Laplacian);

[eigenVect, eigenVal] = eig(Laplacian);
[lambda_sym,index_sym] = sort(diag(eigenVal));

%% Obtaining the gaussian r.d. signal smoothed by the tykhonov regularization
X_0 = randn(m,l);
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

figure('Name','Weight matrix')
surf(W)

figName = strcat('WeightMatrix_',smoothing_method,'.jpg');
saveas(gcf,figName);

figure('Name','Smoothed signal')
surf(X_smooth)

figName = strcat('SmoothedSignal_',smoothing_method,'.jpg');
saveas(gcf,figName);

filename = strcat('SyntheticData_Gaussian',smoothing_method,'.mat');
save(filename,'X_smooth','W','Laplacian','eigenVal','eigenVect','lambda_sym');

% % % %% Generating the starting signal
% % % % The signal is generated from a normal distribution in the range (0,1)
% % % N = 1000;
% % % X_0 = zeros(m,l);
% % %     for i = 1:m
% % %         X_0(i,:) = rand(1,l);
% % %     end
% % %     
% % % %% Constructing the optimization function
% % % 
% % % % For the objective function:
% % % alpha = 10;
% % % X = sdpvar(m,l);
% % % A = (0.5)*(norm(X-X_0))^2;
% % % c = -(norm(X(1,:) - X(1,:))^2)/(2*(sigma^2));
% % % B = (norm(X(1,:)-X(1,:))^2)*exp(c);
% % % 
% % % for i = 1:size(X,1)
% % %     for j = 1:size(X,1)
% % %         if i == 1 && j == 1
% % %             continue;
% % %         else
% % %             c = -(norm(X(i,:) - X(j,:))^2)/(2*(sigma^2));
% % %             B = B + (norm(X(i,:)-X(j,:))^2)*exp(c);
% % %         end
% % %     end
% % % end
% % % 
% % % obj_func = A + (1/alpha)*B;
% % % 
% % % % for the constraints:
% % % F = (X(:,:) >= 0) + (X(:,:) <= 1);
% % % 
% % % %optimization step
% % % diagnostic = optimize(F,obj_func);
% % % 
% % % % Save the smoothed signal
% % % SmoothedSignal = value(X);
    
% % % W = zeros(m);
% % % [W,mn] = random_geometric(sigma,m,X_smooth,W);
% % %     
% % % for big_epoch = 1:1  
% % %     
% % %     % Generating the Laplacian and its decomposition
% % %     
% % %     L = diag(sum(W,2)) - W;
% % %     Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2);
% % %     Lap_rank = rank(Laplacian);
% % %     
% % %     [eigenVect, eigenVal] = eig(Laplacian);
% % %     [lambda_sym,index_sym] = sort(diag(eigenVal));
% % %     X_freq = X_smooth;
% % %     
% % %     % Smoothing the signal
% % %     
% % %     flag = 1;
% % %     switch flag
% % %         case 1
% % %             % Tikhonov regularization
% % %             smoothing_method = 'Tikhonov';
% % %             X_smooth = tikhonov(X_freq,eigenVect,lambda_sym,10,m);
% % %         case 2
% % %             % Heat Kernel regularization
% % %             smoothing_method = 'HeatKernel';
% % %             X_smooth = heat(X_freq,eigenVect,lambda_sym,10,m);
% % %         case 3
% % %             % no regularization
% % %             smoothing_method = 'NoReg';
% % %     end
% % %     
% % %     % Obaining the weight matrix
% % %     
% % %     [W,mn] = random_geometric(sigma,m,X_smooth,W);
% % % end



