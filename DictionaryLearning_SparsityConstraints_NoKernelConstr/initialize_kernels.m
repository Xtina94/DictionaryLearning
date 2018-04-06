function [beta_coefficients] = initialize_kernels(param)
% I have to initialize the beta coefficients in order to have a reduced
% polynomial to learn. I don't have to initialize also the gamma
% coefficients, I have to retrieve those ones from the coefficient
% update step

%Input:
%     percentage = threshold for filtering the kernels. It has to be a value 0<= m <= K with K = degree of polynomial.
%     and indicates the number of eigenvalues that are the roots of our
%     polynomial kernel
%     param-lambda_sym = vector containing all the eigenvalues
       
    %% building up the transpose vandermonde matrix of the eigenvalues
    m = param.percentage;
    vand_eig = zeros(m,m+1);
    img_dim = m+1;
    lambdas = param.lambda_sym;
    
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
    
    %% retrieving the kernel of the Vandermonde matrix
    
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



    
% % %     %% Define the beta vector coefficients
% % %      % the generated beta_vector has the most left coefficient associated
% % %      % to the highest power in the polynomial (x^m + x^(m-1) + ... + 1)
% % %
% % %     beta_coefficients = (ranperm(100,percentage + 1))/100;
% % %
% % %     %% Kernel matrix pursuit
% % %
% % %     rts = roots(beta_coefficients);
% % %
% % %     % Check for real positive roots
% % %     k = 0;
% % %     for i = 1 : length(rts)
% % %         if isreal(rts(i)) && rts(i) > 0 && rts(i) < 2
% % %             k = k+1;
% % %         end
% % %     end
% % %
% % %     chosen_lambda = zeros(1,k);
% % %
% % %     k = 0;
% % %     for i = 1 : length(rts)
% % %         if isreal(rts(i)) && rts(i) > 0 && rts(i) < 2
% % %             k = k+1;
% % %             chosen_lambda(k) = rts(i);
% % %         end
    
% % %     %% kernel generation
% % % 
% % %     kernel_alpha = zeros(param.S,param.N);
% % %     kernel_beta = zeros(param.S,param.N);
% % %     kernel = zeros(param.S,param.N);
% % % 
% % %     D_alpha = cell(4,1);
% % %     D_beta = cell(4,1);
% % %     D = cell(4,1);
% % % 
% % %     for i = 1:param.S
% % %         for j = 1:param.N
% % %             kernel_alpha(i,j) = alpha_vector(i,:)*(param.lambda_powers_alpha{j})';
% % %             kernel_beta(i,j) = beta_coefficients(i,:)*(param.lambda_powers_beta{j})';
% % %             kernel(i,j) = kernel_alpha(i,j)*kernel_beta(i,j);
% % %         end
% % % 
% % %         for k = 0 : K - percentage
% % %             D_alpha{i} =  alpha_vector(i,k+1)*param.Laplacian_powers_alpha{k + 1};
% % %         end
% % % 
% % %         for m = 0 : percentage
% % %             D_beta{i} =  beta_coefficients(i,m+1)*param.Laplacian_powers_beta{m + 1};
% % %         end
% % % 
% % %         D{i} = D_alpha{i}*D_beta{i};
% % %     end
% % % end