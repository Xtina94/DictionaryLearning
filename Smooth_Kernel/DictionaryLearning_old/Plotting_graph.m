% Plots

x = [1:25];
y1 = [0.24
    0.12
    0.11
    0.12
    0.11
    0.11
    0.12
    0.11
    0.12
    0.13
    0.13
    0.14
    0.12
    0.11
    0.12
    0.11
    0.12
    0.13
    0.12
    0.12
    0.12
    0.12
    0.12
    0.12
    0.12];

% y2 = [];

% y3 = [0.08
% 0.08
% 0.09
% 0.08
% 0.08
% 0.06
% 0.07
% 0.07];

avgTime = mean(y1);
xi = 0:0.01:25;
y1 = pchip(x,y1,xi);
% y2 = pchip(x,y2,xi);
% y3 = pchip(x,y3,xi);

figure()
hold on
xlabel('number of the iteration');
ylabel('CPU time per iteration (s)');
plot(xi,y1);
% plot(xi,y2);
% plot(xi,y3);
% legend('y = constraints on kernels', 'y = original algorithm','y = smooth kernel structure','Location','northwest');
hold off

filename = 'CPU time Uber';
saveas(gcf,filename,'bmp');