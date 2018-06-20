function [ original_Laplacian, W, N ] = create_Laplacian_isolated(W)

%% Compute the Laplacian and the normalized Laplacian operator 
L = diag(sum(W,2)) - W; % combinatorial Laplacian
N = size(W,1);
for i=1:N
    ze=1;
    for j=1:N
        if (L(i,j)~=0)
            ze = 0;
        end
    end
    if(ze)
        W(i,i)=1;
    end
end
original_Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian

end

