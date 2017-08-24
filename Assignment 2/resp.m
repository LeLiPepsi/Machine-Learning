%% Responsibility with respect to point x(i) 
%% given parameters theta(priors, means and cov. matrices) 
% Input: x      - point with dimension 2 x 1
%        priors - vector with dimension k x 1
%        means  - means of each cluster with dimension 2 x k
%        covmat - within-class covariance matrices of each cluster with
%                 dimension 2 x 2 x k
% Ouput: p      - vector of responsibilities with dimension k x 1
%        sum_p  - sum of vector p
function [p,sum_p] = resp(x,priors,means,covmat)
    [~,k] = size(means);
    p = zeros(k,1);
    for i = 1 : k
       p(i) = priors(i) * gauss2d(x,means(:,i),covmat(:,:,i)); 
    end
    sum_p = sum(p);
    p = p ./ sum_p;
end

