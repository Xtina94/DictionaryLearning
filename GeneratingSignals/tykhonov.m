function X_smooth = tykhonov(X,eigenVect,eigenVal,alpha)
    h =  1/(1+alpha*eigenVal); %the tykhonov filter
    X_smooth = eigenVect(:,1)'*h(1)*eigenVect(:,1)*X;
    for i = 2:size(X,1) 
        X_smooth = X_smooth + eigenVect(:,i)'*h(i)*eigenVect(:,i)*X;
    end
end