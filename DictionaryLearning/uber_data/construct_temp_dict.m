function [ learned_dictionary, param ] = construct_temp_dict( Laplacian,param )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
for k=0 : max(param.K)
        Laplacian_powers{k + 1} = Laplacian^k;
    end

    for i=1:param.S
        learned_dict{i} = zeros(param.N);
    end

    for k = 1 : max(param.K)+1
        for i=1:param.S
            learned_dict{i} = learned_dict{i} + param.alpha{i}(k)*Laplacian_powers{k};
        end
    end

    learned_dictionary = [learned_dict{1}];
    for i = 2: param.S
            learned_dictionary = [learned_dictionary, learned_dict{i}];
    end
end

