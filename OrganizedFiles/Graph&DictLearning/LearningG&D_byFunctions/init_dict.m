function [initial_dictionary,param] = init_dict(param)
    initial_dictionary = rand(param.N);

    param.lambda_power_matrix = zeros(param.N,max(param.K) + 1);
    param.lambda_power_matrix(:,2) = 1.2*rand(param.N,1);

    for i = 1:max(param.K) + 1
        param.lambda_power_matrix(:,i) = param.lambda_power_matrix(:,2).^(i-1);
    end
end