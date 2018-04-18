clear all
close all
load 'uber_evening_log.txt'
for i=1:15
    our(i)=0;
    for j=1:5
        our(i)=our(i)+uber_evening_log((i-1)*7+j);
    end
    our(i)=our(i)/5;
    geo(i)=uber_evening_log((i-1)*7+6);
end
figure()
hold on
plot(our,'LineWidth',2,'color','red')
plot(geo,'LineWidth',2,'color','blue')
legend({'Learned graph','Geographical graph'},'FontSize',12)
xlabel('Number of atoms','FontSize',16)
ylabel('Signal approximation MRSE','FontSize',16)
