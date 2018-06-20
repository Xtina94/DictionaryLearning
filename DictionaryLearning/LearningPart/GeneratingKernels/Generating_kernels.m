%Generate the polynomial approximation of a sin function thorugh its Talor
%expansion
clear all
close all
path = 'C:\Users\Cristina\Documents\GitHub\DictionaryLearning\LearningPart\GeneratingKernels\';

syms x
g = 1.5*sin(3*x);
g_6 = cos(2*x);
g_7 = -1.3*sin(3.85*x);
g_8 = 1.17*sin(3.5*x);
g_9 = -2*cos(1.75*x);
g_2 = 1.5*sin(3*x) + cos(2*x);
g_3 = -1.3*sin(3*x) - 2.5*cos(2*x);
g_4= 1.17*sin(3.5*x) - 2*cos(1.75*x);
g_5 = 3.5*sin(2.7*x) - 0.55*cos(2.7*x);

% % % t = taylor(g, 'Order', 15);
% % % t_2 = taylor(g_2, 'Order', 15);
% % % t_3 = taylor(g_3, 'Order', 15);
% % % t_4 = taylor(g_4, 'Order', 15);
% % % t_5 = taylor(g_5, 'Order', 15);
% % % 
% % % % t = t + 1.85;
% % % t_2 = (t_2 + 0.2)/1.4;
% % % t_3 = (t_3 + 0.2)/1.4;
% % % t_4 = (t_4 + 0.2)/1.4;
% % % t_5 = (t_5 + 0.2)/1.4;

t = taylor(g, 'Order', 16);
t_2 = taylor(g_2, 'Order', 16);
t_3 = taylor(g_3, 'Order', 16);
t_4 = taylor(g_4, 'Order', 16);
t_5 = taylor(g_5, 'Order', 16);

% t = t + 1.85;
t_2 = (t_2 + 3)/8;
t_3 = (t_3 + 3)/8;
t_4 = (t_4 + 3)/8;
t_5 = (t_5 + 3)/8;

interval = [0,2];
figure('Name','First kernel')
hold on
fplot(t_2,interval)
fplot(t_3,interval)
fplot(t_4,interval)
fplot(t_5,interval)
hold off

filename = strcat(path,'Original kernels.png');
saveas(gcf,filename,'png');

alpha_2 = sym2poly(t_2);
alpha_3 = sym2poly(t_3);
alpha_4 = sym2poly(t_4);
alpha_5 = sym2poly(t_5);

filename = strcat(path,'2HF_kernels');
save(filename,'alpha_3','alpha_4');

filename = strcat(path,'2LF_kernels');
save(filename,'alpha_2','alpha_5');