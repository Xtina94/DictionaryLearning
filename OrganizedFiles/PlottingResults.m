figure('Name','Graphical results')
subplot(2,2,1)
title('X\_norm');
hold on
x = categorical({'Graph L'; 'Dict L smoothness'; 'Dict L no smoothness'; 'G&D no smoothness'; 'G&D smoothness'});
y = [4.2718 20.2937 12.786 36.3058; 23.47 81.76 29.1378 26.1998; 9.2540 28.8372 4.7679 7.6687; 17.006 5.3798 7.6467 21.0606; 71.1964 68.2494 17.892 52.4246];
bar(x,y);
hold off
subplot(2,2,2)
title('Error')
hold on
x = categorical({'Graph L'; 'Dict L smoothness'; 'Dict L no smoothness'; 'G&D no smoothness'; 'G&D smoothness'});
y = [0.1278 0.0746 0.0325 0.201; 0.046974 0.42176 0.11422 0.18519; 0.0063292 0.1175 0.024662 0.073587; 0.1215 0.0702 0.039 0.0883; 0.1884 0.0894 0.0916 0.1983];
bar(x,y);
hold off
subplot(2,2,3)
title('optPrec')
hold on
x = categorical({'Graph L'; 'Dict L smoothness'; 'Dict L no smoothness'; 'G&D no smoothness'; 'G&D smoothness'});
y = [0.0789 0.727 0.8525 0.525; 0 0 0 0; 0 0 0 0; 0.0997 0.6979 0.6854 0.4459; 0.0897 0.7232 0.7143 0.5395];
bar(x,y);
hold off
subplot(2,2,4)
title('optRec')
hold on
x = categorical({'Graph L'; 'Dict L smoothness'; 'Dict L no smoothness'; 'G&D no smoothness'; 'G&D smoothness'});
y = [0.0707000000000000 0.727000000000000 0.852500000000000 0.525000000000000; 0 0 0 0; 0 0 0 0; 0.0969000000000000 0.686000000000000 0.666700000000000 0.437500000000000; 0.0524000000000000 0.713300000000000 0.683100000000000 0.512500000000000];
bar(x,y);
hold off
legend('Dorina','Uber','HeatKernel','DoubleHeat','Location','southeast');
