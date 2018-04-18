%%
clear all
close all
addpath(genpath(pwd))
addpath(genpath('/Users/Hermina/Documents/Graph_SP/installations'));

load uber_data.mat
period = 'weekdays';
if strcmp(period, 'weekends')
    all_days = hist2(:,2:end)';
    weekends = all_days(:,[24*5+1:24*7 24*12+1:24*14 24*19+1:24*21 24*26+1:24*28]);
    norm_col = sum(weekends.^2,1);
    weekends = weekends/sqrt(max(norm_col));
    X = weekends;
elseif strcmp(period, 'weekdays')
    all_days = hist2(:,2:end)';
    diff = setdiff([1:720],[24*5+1:24*7 24*12+1:24*14 24*19+1:24*21 24*26+1:24*28]);
    weekdays = all_days(:, diff);
    norm_col = sum(weekdays.^2,1);
    weekdays = weekdays/sqrt(max(norm_col));
    X = weekdays;
else
    
    X = hist2(:,2:end)';
    norm_col = sum(X.^2,1);
    X = X./sqrt(max(norm_col));
    
end
% select time interval
rush_hours = [];
%start_time = 8;
%end_time = 10;
 
%start_time = 17;
%end_time = 19;
 
%start_time = 11;
%end_time = 16;
 
start_time = 20;
end_time = 24;

%start_time = 1;
%end_time = 7;
 
num_of_days = size(X,2)/24;
for ii = 1 : num_of_days
    rush_hours = [rush_hours weekdays(:,24*(ii-1)+start_time:24*(ii-1)+end_time)];
    X = rush_hours;
end
 
param.N = size(X,1); % number of nodes in the graph
param.S = 2;  % number of subdictionaries 
param.J = param.N * param.S; % total number of atoms 
param.K = [15 15]; % polynomial degree of each subdictionary
param.c = 1; % spectral control parameters
param.epsilon = 0.05;%0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
%kaj ne bi ovaj epsilon gore trebao biti jako velik?
param.mu = 1;%1e-2; % polynomial regularizer paremeter
param.y = X;
param.y_size = size(param.y,2);

%% generate dictionary polynomial coefficients from heat kernel
param.t(1) = 2;
param.t(2) = 1;
param.alpha = generate_coefficients(param);
disp(param.alpha);


%% compute coordinates
for i = 2 : numel(zone_list)
    [rn, ~] = find(strcmp(data2{4},zone_list(i)));
    lat2(i-1,1) = mean(data2{2}(rn));
    lon2(i-1,1) = mean(data2{3}(rn));   
end
 
for i = 2 : numel(zone_list)
    [rn, ~] = find(strcmp(data1{4},zone_list(i)));
    lat1(i-1,1) = mean(data1{2}(rn));
    lon1(i-1,1) = mean(data1{3}(rn));
end
lat = mean([lat1 lat2], 2);
lon = mean([lon1 lon2], 2);


%% construct geographical graphs
figure();
geo = gaussian_knn(lon,lat,6);
wgPlot(geo+diag(ones(param.N,1)),[lon,lat],'vertexWeight',diag(ones(param.N,1)) );
g = plot_google_map('maptype','roadmap');

figure();
geo_gk = gaussian_nedge(lon,lat,4*29);
wgPlot(geo_gk+diag(ones(param.N,1)),[lon,lat],'vertexWeight',diag(ones(param.N,1)) );
g = plot_google_map('maptype','roadmap');


%% initialise learned data

param.T0 = 8;
[param.Laplacian, learned_W] = init_by_weight(param.N);
[learned_dictionary, param] = construct_dict(param);
alpha = 2;
for big_epoch = 1:500
    %% optimise with regard to x
    disp(['Epoch... ',num2str(big_epoch)]);
    x = OMP_non_normalized_atoms(learned_dictionary,param.y, param.T0);
    errors(2*big_epoch-1) = app_error(param.y, learned_dictionary, x);

    %% optimise with regard to W 
    maxEpoch = 2;
    beta = 10^(-2);
    old_L = param.Laplacian;
    [param.Laplacian, learned_W] = update_graph(x, alpha, beta, maxEpoch, param,learned_W, learned_W);
    [learned_dictionary, param] = construct_dict(param);
    errors(2*big_epoch) = app_error(param.y, learned_dictionary, x);
    alpha = alpha*0.985;
end
%%
%figure()
%plot(errors)
%%
uber_graph = treshold_by_edge_number(param.Laplacian, 4*29);
[learned_dictionary, param] = construct_temp_dict(uber_graph,param);
x = OMP_non_normalized_atoms(learned_dictionary,param.y, param.T0);
uber_W = learned_W.*(uber_graph~=0);
figure();
wgPlot(uber_W+diag(ones(param.N,1)),[lon,lat],'vertexWeight',diag(ones(param.N,1)) );
g = plot_google_map('maptype','roadmap');