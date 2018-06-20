function X = generate_sparsity(t0,m,l)
    X = zeros(t0*m,l);
    for i = 1:m
        indexes = randperm(m*t0);
        indexes = indexes(1:4);
        for j = 1:t0
            X(indexes(j),i) = rand;
        end
    end
    
end