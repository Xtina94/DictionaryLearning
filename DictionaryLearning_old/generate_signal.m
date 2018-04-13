function [SampleSignal, initial_dictionary] = generate_signal(output,param)

alpha_coefficients = zeros(param.S,max(param.K)+1);

for s = 1:param.S
    for i = 1:param.K(s)+1
% % %         alpha_coefficients(s,i) = output.alpha((s-1)*param.K(s)+i);
        alpha_coefficients(s,i) = output.coefficients(i,s);
    end
end

%% Generate the dictionary

Dictionary = zeros(param.N,param.S*param.N);
for s = 1 : param.S
    for i = 1 : param.K(s)+1
        Dictionary(:,((s-1)*param.N)+1:s*param.N) = Dictionary(:,((s-1)*param.N)+1:s*param.N) +  alpha_coefficients(s,i) .* param.Laplacian_powers{i};
    end
end

%% Generate the sparsity mx

X = OMP_non_normalized_atoms(Dictionary,param.TrainSignal, param.T0);

%% Obtain the signal

SampleSignal = Dictionary*X;
initial_dictionary = Dictionary;

end