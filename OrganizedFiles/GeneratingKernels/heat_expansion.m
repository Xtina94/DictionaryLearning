function comp_alpha = heat_expansion(param)
    %Generates coefficients for the kernels of polynomial dictionary as a
    %Taylor approximation to heat diffusion.

    for j = 1:param.S
        for i = 0:max(param.K)
            comp_alpha(i+1,j) = ((-param.t(j))^(i))/factorial(i);
            comp_alpha(i+1,j) = (-1)^(j+1)*comp_alpha(i+1,j);
        end
    end
    
    comp_alpha(1,2:2:param.S) = 1;

    %this will invert the graph of the polynomial -> we want to be able to
    %efficiently represent all frequencies of the signal (we're approximating
    %1-heat_kernel here)
% % %     comp_alpha{2}= -comp_alpha{2};
% % %     comp_alpha{2}(1) = comp_alpha{2}(1) + 1;
end
