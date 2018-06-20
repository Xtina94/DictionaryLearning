function output = generate_kernels(param, n_roots)
% Generates the kernels' polynomial coefficients in a random way
% input:
%       degree = polynomial degree
%       n_kernels = number of different kernels
%       lambdas = vector of eigenvalues
%       n_roots = number of lambdas which are roots of the polynomial

S = param.S;
K = max(param.K);
m = n_roots;
N = param.N;

kernels_gamma = zeros(S,K-m+1);
kernels = zeros(S,K+1);
lambda_powers = param.lambda_power_matrix;
lambda_powers_beta = lambda_powers(N-m+1:N,1:m+1);

%% retrieving the kernel of the Vandermonde matrix

betas = null(lambda_powers_beta);

i = 1;

if betas(1,1) > 0
    kernels_beta = betas(:,1);
else
    while betas(1,i) < 0
        if i < length(betas(1,:))
            i = i+1;
        elseif i == length(betas(1,:))
            betas = -betas;
        end
    end
    kernels_beta = (betas(:,i))';
end

%% Generating the gamma coefficients

for i = 1:S
    kernels_gamma(i,:) = rand([1,length(kernels_gamma(i,:))]);
end

kernels_prova = zeros(S,K+1);
for i = 1:S
    for j = 1:length(kernels_prova(i,:))
        kernels_prova(i,j) = ((-1)^(randi(10,1))*rand(1));
    end
end

%% Creating the general kernel vector

for s = 0 : S - 1
    
    %For block A
    for i = 1 : K - m + 1
        for j = 1 : i
            kernels(s+1,i) = kernels(s+1,i) + kernels_gamma(s+1,j)*kernels_beta(i-j+1);
        end
    end
    
    %For block B
    for i = K - m + 2 : m + 1
        for j = 1 : K - m +1
            kernels(s+1,i) = kernels(s+1,i) + kernels_gamma(s+1,j)*kernels_beta(i-j+1);
        end
    end
    
    %For block C
    index = m;
    for i = 2 : K - m + 1
        for j = i : K - m + 1
            kernels(s+1,index+i) = kernels(s+1,index+i) + kernels_gamma(j)*kernels_beta(m-j+i+1);
        end
    end
end
kernels = kernels';

%% Verifying the correctness of the coefficients vector
produc_t = (lambda_powers(:,1:K-m+1)*kernels_gamma');
for s = 1:S
    output.kernels(:,s) = (lambda_powers(:,1:m+1)*kernels_beta).*produc_t(:,s);
end
output.coefficients = kernels;
output.coefficients_beta = kernels_beta;
output.verification = lambda_powers*kernels;


%% Plotting the kernels
figure('Name', 'Original Generated Kernels')
for s = 1:S
    hold on;
    plot(lambda_powers(:,2),output.kernels(:,s));
end
hold off;

figure('Name', 'Verification')
for s = 1:S
    hold on;
    plot(lambda_powers(:,2),output.verification);
end
hold off;
end