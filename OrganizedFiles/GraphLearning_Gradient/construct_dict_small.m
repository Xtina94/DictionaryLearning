function [ learned_dictionary, param ] = construct_dict_small(param)
%construct dictionary and save Laplacian powers
for k=0 : max(param.K)
        param.Laplacian_powers_small{k + 1} = param.Laplacian_small^k;
    end

    for i=1:param.S
        learned_dict{i} = zeros(param.N_small);
    end

    for k = 1 : max(param.K)+1
        for i=1:param.S
            learned_dict{i} = learned_dict{i} + param.alpha{i}(k)*param.Laplacian_powers_small{k};
        end
    end

    learned_dictionary = [learned_dict{1}];
    for i = 2: param.S
            learned_dictionary = [learned_dictionary, learned_dict{i}];
    end
    
    % Remove the 1 in the columns that aare not used to construct de signal
    for i = 1:param.N
        if isempty(find(param.sources == i))
            learned_dictionary(:,i) = 0;
        end
    end
    
end

