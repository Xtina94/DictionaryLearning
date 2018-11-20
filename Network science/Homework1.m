clear all;
close all;

% Load the dataset in example
load bioCel.txt

%% Extract the number of nodes and links
nodes_firstCol = sort(bioCel(:,1));
nodes_secondCol = sort(bioCel(:,2));
nodes = [nodes_firstCol, nodes_secondCol];
nodes = unique(nodes);
n_nodes = length(nodes); % number of nodes

degree = zeros(n_nodes,2); % a n_nodes x 2 matrix with each row having nodeName --> nodeDegree
for i = 1:n_nodes
    degree(i,:) = [nodes(i), histc(bioCel(:,1),nodes(i))];
end

n_links = sum(degree(:,2));

%% Extract the PDF and the CCDM measure
A = sparse(bioCel(:,2),bioCel(:,1),ones(size(bioCel,1),1),n_nodes,n_nodes);

figure('Name','Sparse representation of the adjacency matrix')
spy(A)

% Linear PDF plot
k = unique(degree(:,2)); % all the degree types
M = length(k);
pdf = zeros(M,2); % a n_nodes x 2 matrix with each row having degree --> #Nodes with that degree

for i = 1:M
    pdf(i,:) = [k(i), histc(degree(:,2),k(i))];
end
pdf(:,2) = pdf(:,2)./n_links;

figure('Name','Linear pdf plot')
scatter(pdf(:,1),pdf(:,2));

%PDF logarithmic plot
figure('Name','Logarithmic pdf plot')
loglog(pdf(:,1),pdf(:,2),'oblue');

% % % %PDF log-binning plot
% % % degree = degree(3:length(degree));
% % % BinNum = 25; %Like professor's chart
% % % [midpoints,Freq,eFreq] = lnbin(degree, BinNum);
% % % 
% % % figure('Name','Log-binning plot')
% % % loglog(midpoints,Freq,'o');
% % % % axis([0 250 0 0.05]);

% % % %% PDF Logarithmic CCDM plot
% % % gamma = 3.5;
% % % k = k(3:length(k));
% % % M = length(k);
% % % ccdm_pdf_35 = zeros(1,M);
% % % for i = 1:M
% % %     ccdm_pdf_35(i) = k(i)^(1-gamma);
% % % end
% % % 
% % % %With ML estimation
% % % k_min = min(k);
% % % den = log(degree(1)/k_min);
% % % i=2;
% % % while i <= N-2
% % %     if degree(i)==0
% % %         i = i+1;
% % %     else
% % %         my_log(i) = log(degree(i)/k_min);
% % % %     den = den + log(degree(i)/k_min);
% % %     end
% % %     i = i+1;
% % % end
% % % 
% % % den = sum(my_log);
% % % 
% % % gamma = 1 + sum(N-2)/den;
% % % ccdm_pdf = zeros(1,M);
% % % for i = 1:M
% % %     ccdm_pdf(i) = k(i)^(1-gamma);
% % % end
% % % 
% % % %NOTA sistema la questione dell'individuazione di gamma
% % % figure('Name','Logarithmic CCDM_35')
% % % loglog(k,ccdm_pdf_35,'o');
% % % 
% % % figure('Name','Logarithmic CCDM')
% % % loglog(k,ccdm_pdf,'o');
