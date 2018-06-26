function [out] = obtain_laplacian(W,param,alpha)
out_L = diag(sum(W,2)) - W; % combinatorial Laplacian
Laplacian = (diag(sum(W,2)))^(-1/2)*out_L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian
[eigenMat, eigenVal] = eig(Laplacian);
[lambdaSym,indexSym] = sort(diag(eigenVal));
lambdaPowerMx(:,2) = lambdaSym;
for i = 1:max(param.K) + 1
    lambdaPowerMx(:,i) = lambdaPowerMx(:,2).^(i-1);
    Laplacian_powers{i} = Laplacian^(i-1);
end

ker = zeros(param.N,param.S);
for i = 1 : param.S
    for n = 1:param.N
        ker(n,i) = ker(n,i) + lambdaPowerMx(n,:)*alpha(:,i);
    end
end

out.ker = ker;
out.Laplacian = Laplacian;
out.lambdaSym = lambdaSym;
out.lambdaPowerMx = lambdaPowerMx;
out.eigenMat = eigenMat;
out.Laplacian_powers = Laplacian_powers;
end