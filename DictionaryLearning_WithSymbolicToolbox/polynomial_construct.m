function alpha_coefficients = polynomial_construct(param, gammaCoeff)

    K = max(param.K);
    m = param.percentage;
    q = sum(param.K)+param.S;
    alpha_coefficients = sdpvar(q,1);
    gamma_coefficients = gammaCoeff;
%     alpha_coefficients(1,1) = gamma_coefficients(1)*param.beta_coefficients(1);
    
    for s = 0 : param.S - 1

        %For block A
        for i = 1 : K - m + 1
            alpha_coefficients(i+s*(param.K(s + 1)+1),1) = gamma_coefficients(1)*param.beta_coefficients(i-1+1);
            for j = 2 : i
                alpha_coefficients(i+s*(param.K(s + 1)+1),1) = alpha_coefficients(i+s*(param.K(s + 1)+1),1) + gamma_coefficients(j)*param.beta_coefficients(i-j+1);
            end
        end

        %For block B
        for i = K - m + 2 : m + 1
            alpha_coefficients(i+s*(param.K(s + 1)+1),1) = gamma_coefficients(1)*param.beta_coefficients(i-1+1);
            for j = 2 : K - m +1
                alpha_coefficients(i+s*(param.K(s + 1)+1),1) = alpha_coefficients(i+s*(param.K(s + 1)+1),1) + gamma_coefficients(j)*param.beta_coefficients(i-j+1);
            end
        end

        %For block C
        index = m;
        for i = 2 : K - m + 1
            alpha_coefficients(index+i+s*(param.K(s + 1)+1),1) = gamma_coefficients(1)*param.beta_coefficients(m+1);
            for j = i+1 : K - m + 1
                alpha_coefficients(index+i+s*(param.K(s + 1)+1),1) = alpha_coefficients(index+i+s*(param.K(s + 1)+1),1) + gamma_coefficients(j)*param.beta_coefficients(m-j+i+1);
            end
        end
    end
end