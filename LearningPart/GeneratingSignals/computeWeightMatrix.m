function  W  = computeWeightMatrix( Points, sigma, threshw, varargin )
%computeWeightMatrix calculate weight btw each pair of pts
%   Remove the edges whose weight under the threshw
%   Calculate the weights in terms of Euclidean distance using a Gaussian
%   function with sigma
%   last param: 1, plot the weight distribution

func_rbf = @(x) exp(-x.^2/(2*sigma^2));
% Calcualte the Euclidean distance btw each pair of points
Dist = pdist2(Points,Points,'euclidean');
if (nargin == 4)
    plot(sort(Dist(:)),func_rbf(sort(Dist(:))));
    hold on
    plot(sort(Dist(:)),repmat(threshw,numel(Dist),1),'r');
end
W = func_rbf(Dist);
% remove the edges having a weight under the threshold
W(W < threshw) = 0;
% make diagonal zero
W(logical(eye(size(W)))) = 0;
end

