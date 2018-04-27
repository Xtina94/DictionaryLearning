function [X,W,mn] = random_geometric(sigma, threshold, m, l)
X = zeros(l,m);
    for i = 1:m
        X(:,i) = rand(l,1);
    end
    W = zeros(m,m);
    d = zeros(1,l);
    
    Dist = pdist2(X,X,'euclidean');
    func_rbf = @(x) exp(-x.^2/(2*sigma^2));
    
    mn = zeros(m,1);
    for i = 1:m
        for j = 1:m
            d(1,:) = (X(i,:) - X(j,:)).^2;
            di = sqrt(sum(d));
            c = -(di.^2)/(2*(sigma^2));
            W(i,j) = func_rbf(Dist(i,j));
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