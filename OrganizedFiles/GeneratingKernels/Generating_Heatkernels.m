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

for i = 1:param.S
    eval(strcat('kernel_',num2str(i),'(x) = x^(0)*param.alpha(1,',num2str(i),')'));
end

for i = 1:param.S
    for j = 2:deg+1
        eval(strcat('kernel_',num2str(i),'(x) = kernel_',num2str(i),...
            '(x) + x^(',num2str(j),'-1)*param.alpha(',num2str(j),',1)'));
    end
end

load comp_lambdaSym.mat;
for i = 1:param.S
    eval(strcat('kernels(:,',num2str(i),') = kernel_',num2str(i),'(lambdas)'));
end

figure('Name','Heat kernels representation')
hold on
plot(lambdas,kernels);
hold off

%% Save results to file

filename = [path,'LF_HeatKernel_plot.png'];
saveas(gcf,filename);

LF_HeatKernel = param.alpha;
filename = [path,'LF_HeatKernel.mat'];
save(filename,'LF_HeatKernel');



