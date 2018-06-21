function [learned_dictionary, param ] = construct_dict( param )
%construct dictionary and save Laplacian powers

    K = max(param.K);
    [param.eigenMat, param.eigenVal] = eig(param.Laplacian);
    param.lambda_sym = sort(diag(param.eigenVal));

    for k = 0 : K
        param.Laplacian_powers{k + 1} = param.Laplacian^k;
    end

    param.lambda_powers = zeros(length(param.lambda_sym),K+1);
    for k = 0 : K
        param.lambda_powers(:,k+1) = param.lambda_sym.^k;
    end

    for i = 1:param.S
        learned_dict{i} = zeros(param.N);
    end

    for k = 1 : K+1
        for i=1:param.S
            learned_dict{i} = learned_dict{i} + param.alpha((i-1)*K + k)*param.Laplacian_powers{k};
        end
    end

    learned_dictionary = [learned_dict{1}];
    for i = 2: param.S
        learned_dictionary = [learned_dictionary, learned_dict{i}];
    end
end

