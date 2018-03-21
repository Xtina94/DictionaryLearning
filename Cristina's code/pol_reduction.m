%Trial code for kernel factorization
%Cristina Gava

function [betas, rts] = pol_reduction(lambdas,m)

% load lambdas.mat

%Input:
%     m = threshold for filtering the kernels. It has to be a value 0<= m <= K with K = degree of polynomial.
%         and indicates the number of eigenvalues that are the roots of our
%         polynomial kernel
%     lambdas = vector containing all the eigenvalues
       
%building up the transpose vandermonde matrix of the eigenvalues
vand_eig = zeros(m,m+1);
img_dim = m+1;

while img_dim >= m
    rts =  lambdas(length(lambdas)-m+1:length(lambdas),1);
    for i = 1:m+1
        for j = 1:m
            vand_eig(j,i) = rts(j)^(i-1);
        end
    end
    m = m+1;
    img_dim = rank(vand_eig);
end

m = m - 1;

betas = null(vand_eig);
%beta = betas(:,1);
    