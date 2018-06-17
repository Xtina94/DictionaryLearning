function [W,mn,variance] = random_geometric(sigma,m,X,k)
    W = zeros(m,m);
    for i = 1:m
        for j = 1:m
            c = -(norm(X(i,:) - X(j,:))^2)/(2*(sigma^2));
            W(i,j) = exp(c);
            if j == i
                W(i,j) = 0;
            end
        end
    end
    
    mn = mean(W,2);
    variance = zeros(m,1);
    for i = 1:m
        variance(i) = var(W(i,:));
    end
    
    values = W(1,2);
    for i = 1:m
        for j = 1:m
            if j <= i || (i == 1 && j == 2)
                continue;
            else
                values = [values W(i,j)];
            end
        end
    end
    
    values = sort(values);
    counter = 1;
    x_axis = -1;
    for i = 1:length(values)-1
        if values(i+1) == values(i)
            counter = counter + 1;
        else
            if x_axis == -1 %I'm in the first occurnce count
                x_axis = values(i);
                y_axis = counter;
            else
                x_axis = [x_axis values(i)];
                y_axis = [y_axis counter];
            end
            counter = 1;
        end
    end
    
    figure('Name','Gaussian distribution of the elements of W')
    plot(x_axis,y_axis)
    
    % Take only the s% of the values
    index = floor(length(x_axis)*k);
    thr = values(index);
 
    for i = 1:m
        for j = 1:m
            if W(i,j) < thr
                W(i,j) = 0;
            end
        end
    end
end