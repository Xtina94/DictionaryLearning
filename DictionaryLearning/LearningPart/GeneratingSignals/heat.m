function X_smooth = heat(X,eigenVect,lambda_sym,alpha,m)
    h = zeros(1,m);
    for i = 1:m
        h(i) =  exp(-alpha*lambda_sym(i)); %the tykhonov filter
    end
    
    X_smooth = eigenVect(:,1)'*h(1)*eigenVect(:,1);
    for i = 2:size(X,1) 
        X_smooth = X_smooth + eigenVect(:,i)'*h(i)*eigenVect(:,i);
    end
    X_smooth = X_smooth*X;
end