function X_smooth = tykhonov(X,eigenVect,eigenVal,alpha,m)
    h = zeros(1,m);
    for i = 1:m
        h(i) =  1/(1+eigenVal(i)*alpha); %the tykhonov filter
    end
    
    X_smooth = eigenVect(:,1)'*h(1)*eigenVect(:,1);
    for i = 2:size(X,1) 
        X_smooth = X_smooth + eigenVect(:,i)'*h(i)*eigenVect(:,i);
    end
    X_smooth = X_smooth*X;
end