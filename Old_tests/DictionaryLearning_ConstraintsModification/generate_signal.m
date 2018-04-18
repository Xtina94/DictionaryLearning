function SampleSignal = generate_signal(output, param)

alpha_coefficients = output.coefficients';

%% Generate the dictionary

Dictionary = zeros(param.N,param.S*param.N);
for s = 1 : param.S
    for i = 1 : param.K(s)+1
        Dictionary(:,((s-1)*param.N)+1:s*param.N) = Dictionary(:,((s-1)*param.N)+1:s*param.N) +  alpha_coefficients(s,i) .* param.Laplacian_powers{i};
    end
end

%% Generate the sparsity mx

TrainSignal = zeros(100,600);
X = sparsity_matrix_initialize(param,TrainSignal);

%% Obtain the signal

SampleSignal = Dictionary*X;
end