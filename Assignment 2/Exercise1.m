dbstop if error
load('dataGMM.mat'); % x with dimension m x n
x = Data;
%% Initialization
abs_err = 1e-6;
%% Initialization using k-means algorithm
k = 4;
[m,n] = size(x);
[IDX,C] = kmeans(x',k);
% Initialization of priors
priors = zeros(k,1);
for i = 1 : k
   ni = find(IDX == ones(n,1) * i);
   priors(i) = size(ni,1) / n;
end
% Initialization of means
means = C';% means with dimension m x k
% Initialization of covariance matrices
covmat = zeros(m,m,k);
for i = 1 : k
   dataj = x(:,find(IDX == ones(n,1) * i));
   covmat(:,:,i) = cov(dataj'); 
end
% Initialization of error
l_old = 0;
l_new = log_like(x,priors,means,covmat);
err = abs(l_new);
%% Repeat until log-likelihood converges
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
    miu = zeros(m,k);
    sigmat = zeros(m,m,k);
    for i = 1 : k
        % Update of means
        for j = 1 : n
            miu(:,i) = miu(:,i) + resp_mat(i,j) * x(:,j);
        end
        miu(:,i) = miu(:,i) / n_k(i);
        % Update of covariance matrices
        for j = 1 : n
            sigmat(:,:,i) = sigmat(:,:,i) + resp_mat(i,j) * (x(:,j) - miu(:,i)) * (x(:,j) - miu(:,i))';
        end
        sigmat(:,:,i) = sigmat(:,:,i) / n_k(i);
    end
    means = miu;
    covmat = sigmat;
    %% Evaluate the log-likelihood
    l_new = log_like(x,priors,means,covmat);
    err = abs(l_new - l_old);
end
%% Display learned priors, means and covariance matrices
fprintf('Learned priors are\n');
disp(priors);
fprintf('Learned means are\n');
disp(means);
fprintf('Learned covariance matrices are\n');
disp(covmat);


