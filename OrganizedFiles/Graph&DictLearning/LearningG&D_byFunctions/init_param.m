function [A,B,S,epsilon,degree,N,ds,ds_name,percentage,thresh] = init_param(flag,TrainSignal)

% Adding the paths
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\Comparison_datasets\'); %Folder containing the copmarison datasets
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\'); %Folder containing the training and verification dataset

switch flag
    case 1 %Dorina
        A = TrainSignal;
        B = 20;
        S = 4;  % number of subdictionaries         
        epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 20;
        N = 100; % number of nodes in the graph
        ds = 'Dataset used: Synthetic data from Dorina';
        ds_name = 'Dorina';
        percentage = 15;
        thresh = percentage+60;
    case 2 %Cristina
        A = TrainSignal;
        B = 15;
        S = 2;  % number of subdictionaries        
        epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
        N = 100; % number of nodes in the graph
        ds = 'Dataset used: data from Cristina';
        ds_name = 'Cristina'; 
        percentage = 8;
        thresh = percentage+60;
    case 3 %Uber
        A = TrainSignal;
        B = 15;
        S = 2;  % number of subdictionaries 
        epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
        N = 29; % number of nodes in the graph
        ds = 'Dataset used: data from Uber';
        ds_name = 'Uber';
        percentage = 8;
        thresh = percentage+6;
    case 4 %Heat kernel
        A = TrainSignal;
        B = 15;
        S = 1;  % number of subdictionaries 
        epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
        N = 100; % number of nodes in the graph
        ds = 'Dataset used: data from Heat kernel';
        ds_name = 'Heat';
        percentage = 8;
        thresh = percentage+6;
end

end