%Generate the polynomial approximation of a sin function thorugh its Talor
%expansion
clear all
close all
path = 'C:\Users\Cristina\Documents\GitHub\OrganizedFiles\Datasets\';

param.S = 4;
deg = 15;
param.K = deg*ones(1,param.S);
syms x;

%% generate dictionary polynomial coefficients from heat kernel

for i = 1:param.S
    if mod(i,2) ~= 0
        param.t(i) = 2; %heat kernel coefficients
    else
        param.t(i) = 1; % Inverse of the heat kernel coefficients
    end
end

param.alpha = heat_expansion(param);
syms kernel1 kernel2 kernel3 kernel4;

    kernel1(x) = x^(0)*param.alpha(1,1);
% % %     kernel2(x) = x^(0)*param.alpha(1,2);
% % %     kernel3(x) = x^(0)*param.alpha(1,3);
% % %     kernel4(x) = x^(0)*param.alpha(1,4);
    
    for j = 2:deg+1
        kernel1(x) = kernel1(x) + x^(j-1)*param.alpha(j,1);
% % %         kernel2(x) = kernel2(x) + x^(j-1)*param.alpha(j,2);
% % %         kernel3(x) = kernel3(x) + x^(j-1)*param.alpha(j,3);
% % %         kernel4(x) = kernel4(x) + x^(j-1)*param.alpha(j,4);
    end
    
    kernels(:,1) = kernel1(0:0.0001:2);
% % %     kernels(:,2) = kernel2(0:0.0001:2);
% % %     kernels(:,3) = kernel3(0:0.0001:2);
% % %     kernels(:,4) = kernel4(0:0.0001:2);

figure('Name','Heat kernels representation')
hold on
plot(0:0.0001:2,kernels);
hold off

%% Save results to file

filename = [path,'LF_heatKernel_plot.png'];
saveas(gcf,filename);

LF_heatKernel = param.alpha(:,1);
filename = [path,'LF_heatKernel.mat'];
save(filename,'LF_heatKernel');



