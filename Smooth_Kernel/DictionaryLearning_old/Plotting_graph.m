% Plots

x = [1,2,3,4,5,6,7,8];
y1 = [0.13
0.12
0.14
0.13
0.11
0.14
0.12
0.17];

y2 = [0.13
0.12
0.13
0.16
0.13
0.13
0.14
0.16];

y3 = [0.08
0.08
0.09
0.08
0.08
0.06
0.07
0.07];


xi = 0:0.01:8;
y1 = pchip(x,y1,xi);
y2 = pchip(x,y2,xi);
y3 = pchip(x,y3,xi);

figure()
hold on
xlabel('number of iteration');
ylabel('CPU time per iteration');
plot(xi,y1);
plot(xi,y2);
plot(xi,y3);
legend('y = constraints on kernels', 'y = original algorithm','y = smooth kernel structure','Location','northwest');
hold off

filename = 'CPU time Dorina';
saveas(gcf,filename,'bmp');