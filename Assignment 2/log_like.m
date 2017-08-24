%% log-likelihood function with respect to whole dataset x 
%% given parameters theta(priors, means and cov. matrices)
% Input: x      - dataset(n samples, each sample with dimension 2 x 1)
%        priors - vector with dimension k x 1
%        means  - means of each cluster with dimension 2 x k
%        covmat - within-class covariance matrices of each cluster with
%                 dimension 2 x 2 x k
% Ouput: log_likelihood
function log_likelihood = log_like(x,priors, means, covmat)
    [~,n] = size(x);
    t = 0;
    log_likelihood = 0;
    for i = 1 : n
        [~,t] = resp(x(:,i),priors,means,covmat);
        log_likelihood = log_likelihood + log(t);
    end
end