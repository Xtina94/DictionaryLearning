% % % ds_name_1 = 'Dorina';
% % % ds_name_2 = 'Heat';
% % % ds_name_3 = 'DoubleHeat';
% % % path = 'C:\Users\Cristina\Documents\GitHub\GraphLearningSparsityPriors\Results\05.07.18\';
% % % file_name = '\Output_Norm_PrecRec.mat';
% % % for i = 1:3
% % %     eval(file_,num2str(i),' = ',num2str(path),'ds_name_',num2str(i),num2str(file_name)));
% % % end
figure('Name','Graphical results')
subplot(2,2,1)
title('X\_norm');
hold on
x = categorical({'Graph L'; 'Dict L smoothness'; 'Dict L no smoothness'; 'G&D no smoothness'; 'G&D smoothness'});
y = [7.117612294895877 17.220504249852613 17.44504195400573; 23.47 29.1378 26.1998; 9.2540 4.7679 7.6687; 17.006 7.6467 21.0606; 71.1964 17.892 52.4246];
bar(x,y);
hold off
subplot(2,2,2)
title('Error')
hold on
x = categorical({'Graph L'; 'Dict L smoothness'; 'Dict L no smoothness'; 'G&D no smoothness'; 'G&D smoothness'});
y = [0.1237173830600453 0.027829547538260963 0.11070970420570186; 0.046974 0.11422 0.18519; 0.0063292 0.024662 0.073587; 0.1215 0.039 0.0883; 0.1884 0.0916 0.1983];
bar(x,y);
hold off
subplot(2,2,3)
title('optPrec')
hold on
x = categorical({'Graph L'; 'Dict L smoothness'; 'Dict L no smoothness'; 'G&D no smoothness'; 'G&D smoothness'});
y = [0.09627329192546584 0.9192546583850931 0.536144578313253; 0 0 0; 0 0 0; 0.0997 0.6854 0.4459; 0.0897 0.7143 0.5395];
bar(x,y);
hold off
subplot(2,2,4)
title('optRec')
hold on
x = categorical({'Graph L'; 'Dict L smoothness'; 'Dict L no smoothness'; 'G&D no smoothness'; 'G&D smoothness'});
y = [0.08115183246073299 0.9024390243902439 0.5297619047619048; 0 0 0; 0 0 0; 0.0969000000000000 0.666700000000000 0.437500000000000; 0.0524000000000000 0.683100000000000 0.512500000000000];
bar(x,y);
hold off
legend('Dorina','HeatKernel','DoubleHeat','Location','southeast');
