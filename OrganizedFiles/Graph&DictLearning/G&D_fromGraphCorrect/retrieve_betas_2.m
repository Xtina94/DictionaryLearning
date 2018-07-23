function [beta_coefficients, rts] = retrieve_betas_2(param)
% I have to initialize the beta coefficients in order to have a reduced
% polynomial to learn. I don't have to initialize also the gamma
% coefficients, I have to retrieve those ones from the coefficient
% update step

%Input:
%     percentage = threshold for filtering the kernels. It has to be a value 0<= m <= K with K = degree of polynomial.
%     and indicates the number of eigenvalues that are the roots of our
%     polynomial kernel
%     param-lambda_sym = vector containing all the eigenvalues
          
heat_k = param.heat_k;
    %% retrieving the kernel of the Vandermonde matrix
    if heat_k
        comp_ker = param.startingKer;
        for i = 1 : param.S
            for n = 1:param.N
                comp_ker(n,i) = comp_ker(n,i) + param.lambda_power_matrix(n,:)*param.alpha{i};
            end
        end
        r = comp_ker(param.N-param.percentage:param.N,1);
        % building up the transpose vandermonde matrix of the eigenvalues
        m = param.percentage;
        vand_eig = zeros(m+1,m+1);
        lambdas = param.lambda_sym;
        rts =  lambdas(length(lambdas)-m:length(lambdas),1);
        for i = 1:m+1
            vand_eig(:,i) = rts.^(i-1);
        end
        beta_coefficients = inv(vand_eig)*r;
    else
        % building up the transpose vandermonde matrix of the eigenvalues
        m = param.percentage;
        vand_eig = zeros(m,m+1);
        img_dim = m+1;
        lambdas = param.lambda_sym;
        
        while img_dim >= m
            rts =  lambdas(length(lambdas)-m+1:length(lambdas),1);
            for i = 1:m+1
                vand_eig(:,i) = rts.^(i-1);
            end
            m = m+1;
            img_dim = rank(vand_eig);
        end
        
        m = m - 1;
        betas = null(vand_eig);
        i = 1;
        
        if betas(1,1) > 0
            beta_coefficients = betas(:,1);
        else
            while betas(1,i) < 0
                if i < length(betas(1,:))
                    i = i+1;
                elseif i == length(betas(1,:))
                    betas = -betas;
                end
            end
            beta_coefficients = (betas(:,i));
        end
    end
end