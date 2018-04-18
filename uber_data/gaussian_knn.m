function [ W_knn ] = gaussian_knn( XCoords, YCoords, nn)
    s = 0.05; %sigma in the gaussian
    % CHANGED THIS from Distanz (not recognised) to pdist
    coords = [XCoords(:), YCoords(:)];
    d = squareform(pdist(coords));

    W = exp(-d.^2/(2*s^2)); 
    W = 0.5*(W+W');
    W = W-diag(diag(W));
    W_knn = zeros(size(W,1));
    for i=1:29
        [sorted_col,ind] = sort(W(:,i),'descend');
        kept_ind = ind(1:nn);
        kept_val = sorted_col(1:nn);
        for j=1:nn
            W_knn(kept_ind(j),i)=kept_val(j);
            W_knn(i,kept_ind(j))=kept_val(j);
        end
    end
    disp(sum(sum(W_knn~=0)));
end

