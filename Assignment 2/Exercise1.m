x = load('dataGMM.mat');
%% Initialization
abs_err = 1e-6;
%% Initialization using k-means algorithm
k = 4;
[~,n] = size(x);
[IDX,C,sumd] = kmeans(x',k);
priors = ;
means = ;
covmat = ;
l_old = 0;
l_new = log_like(x,priors,means,covmat);
err = abs(l_new);
%% Repeat until log-likelihood conveges
while err > abs_err
    l_old = l_new;
    %% E-step
     % responsibilities matrix resp_mat with dimension k x n
     resp_mat = [];
     for i = 1:n
        [p,~] = resp(x(:,i),priors,means,covmat);
        resp_mat = [resp_mat p];
     end
    %% M-step
    n_k = sum(resp_mat,2);
    priors = n_k / n;
    miu = zeros(2,k);
    sigmat = zeors(2,2,k);
    for i = 1 : k
        for j = 1 : n
            miu(:,i) = miu(:,i) + resp_mat(i,j) * x(:,j)
        end
        miu(:,i) = miu(:,i) / n_k(i)
        for j = 1 : n
            sigmat(:,:,i) = sigmat(:,:,i) + resp_mat(i,j) * (x(:,j) - miu(:,i)) * (x(:,j) - miu(:,i))';
        end
        sigmat(:,:,i) = sigmat(:,:,i) / n_k(i);
    end
    %% Evaluate the log-likelihood
    l_new = log_like(x,priors,means,covmat);
    err = abs(l_new - l_old);
end
%% Display learned priors, means and covariance matrices



