function [ W ] = gaussian_nedge( XCoords, YCoords, nedges )
s = 0.05; %sigma in the gaussian
% CHANGED THIS from Distanz (not recognised) to pdist
coords = [XCoords(:), YCoords(:)];
d = squareform(pdist(coords));
%d(find(d<0.3))=1;
%d=d-0.18;

%d(find(d<0.3|d>0.7))=1;
%d=d-0.18;

W = exp(-d.^2/(2*s^2)); 
W = 0.5*(W+W');
W = W-diag(diag(W));
[row,column,val] = find(W);
val = sort(val,'descend');
T=val(nedges*2);
W(W<T) = 0; % Thresholding to have sparse matrix

end

