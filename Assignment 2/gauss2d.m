%% 2D-Gaussian Distribution Density Function
% Input: x  - 2 dimensional column vector
%        mean - mean vecter of gaussian distribution
%        covmat - 2 x 2 covariance matrix
%Output: density 

function density = gauss2d(x,mean,covmat)
    [m,~] = size(x);
    density = 1 / sqrt(((2*pi)^m) * det(covmat)) * exp(-0.5*(x-mean)'*covmat*(x-mean));
end