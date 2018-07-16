function [W,Laplacian] = random_geometric(sigma,m,X,k)
    W = zeros(m,m);
    for i = 1:m
        for j = 1:m
            c = -(norm(X(i) - X(j))^2)/(2*(sigma^2));
            W(i,j) = exp(c);
            if j == i
                W(i,j) = 0;
            end
        end
    end
    
    for i = 1:m
        for j = 1:m
            if W(i,j) < k
                W(i,j) = 0;
            end
        end
    end
    
L = diag(sum(W,2)) - W; % combinatorial Laplacian
Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian
end