%Trial code for kernel factorization
%Cristina Gava

function [beta, rts] = pol_reduction(lambdas,m)

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

    i = 1;
          
    if betas(1,1) > 0
        beta = betas(:,1);
    else
        while betas(1,i) < 0
          if i < length(betas(1,:))
              i = i+1;
          elseif i == length(betas(1,:))
              betas = -betas;
          end
        end
        beta = (betas(:,i));
    end
    