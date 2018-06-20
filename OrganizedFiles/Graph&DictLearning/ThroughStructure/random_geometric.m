function [X,W,mn] = random_geometric(sigma, threshold, m, l)
X = zeros(m,l);
    for i = 1:m
        X(i,:) = rand(1,l);
    end
    W = zeros(m,m);
    d = zeros(1,l);
    
    mn = zeros(m,1);
    for i = 1:m
        for j = 1:m
            c = -(norm(X(i,:) - X(j,:))^2)/(2*(sigma^2));
            W(i,j) = exp(c);
            if W(i,j) < threshold
                W(i,j) = 0;
            end
            if j == i
                W(i,j) = 0;
            end
        end
        mn(i) = mean(W(i,:));
    end
end