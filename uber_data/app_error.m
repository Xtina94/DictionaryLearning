function [ error ] = app_error( Y, Dictionary, x )
error = sqrt(sum(sum((Y-Dictionary * x).^2))/numel(Y));
end

