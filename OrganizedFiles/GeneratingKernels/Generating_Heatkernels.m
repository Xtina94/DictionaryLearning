%Generate the polynomial approximation of a sin function thorugh its Talor
%expansion
clear all
close all
path = 'C:\Users\Cristina\Documents\GitHub\OrganizedFiles\GeneratingKernels\Results\';

param.S = 1;
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

temp = heat_expansion(param);

for i = 1:param.S
    param.alpha(:,i) = temp{i};
end

syms kernel;

kernel1(x) = x^(0)*param.alpha(1,1);

for j = 2:deg+1
    kernel1(x) = kernel1(x) + x^(j-1)*param.alpha(j,1);
end

kernels(:,1) = kernel1(0:0.0001:2);

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



