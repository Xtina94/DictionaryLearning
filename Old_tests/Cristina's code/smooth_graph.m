%% Creation of a smooth graph
% Create a W matrix such that the resulting graph signal is smooth
% Input:
%       TrainSignal: the signal over the graph (Nx600 matrix)
%       L = the graph Laplacian (NxN matrix)
%

function smth_signal = smooth_graph(param, TrainSignal)
        smth_signal = smooth2a(TrainSignal, param.K(1), param.K(1));
        
% % %     % Defining the matrix variable for the signal
% % %     param.samples = 600;
% % %     %Y = sdpvar(param.N,param.samples);
% % %     f = sdpvar(param.N,param.samples);
% % %     c = 1;
% % %     smth_signal = zeros(param.N,param.samples);
% % % 
% % %     % Define the objective function
% % %     for i = 1:param.N
% % %         for j = 1:param.N
% % %             if W(i,j)~= 0
% % %                 f(i,:) = f(i,:) + W(i,j)*(f(i,:) - f(j,:)).^2;
% % %             end
% % %         end
% % %         f(i,:) = 0.5*f(i,:);
% % %     end
% % % 
% % %     for s = 1:param.samples
% % %         X = f(:,s);
% % %         
% % %         % Define constraints: let's make the hp that the signals are normalized
% % %         F = lmi((f(:,s) <= c*ones(param.N,1)) + (-f(:,s) <= 0*ones(param.N,1)));
% % %         
% % %         % Optimize
% % %         optimize(F,X,sdpsettings('solver','sdpt3'));
% % %         
% % %         smth_signal(:,s) = X;
% % %     end
end

