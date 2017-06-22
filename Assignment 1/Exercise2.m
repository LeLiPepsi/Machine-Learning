function [d_opt, err_opt, confusion_mat] = Exercise2(d_max)
    images = loadMNISTImages('train-images.idx3-ubyte');
    labels = loadMNISTLabels('train-labels.idx1-ubyte');
    img_test = loadMNISTImages('t10k-images.idx3-ubyte');
    lab_test = loadMNISTLabels('t10k-labels.idx1-ubyte');
    [w n] = size(images);
    [wt nt] = size(img_test);
    % initialization
    d_opt = 0;
    err_opt = 0;
    err_d = zeros(1,d_max);
    confusion_mat = [];
    I = zeros(nt,d_max);
    % compute the mean
    miu = mean(images,2);
    % replace x by x-miu, to obatain zero mean data
    images = images - repmat(miu,1,n);
    img_test = img_test - repmat(miu,1,nt);
    % calculate the covariance matrix of the zero mean data 
    S = cov(images');
    % calculate eigenvalues and eigenvectors of the covariance matrix S
    %e = eig(S);
    [V,D] = eig(S);
    e = diag(D);
    sortmat = [e V'];
    sorted = sortrows(sortmat,'descend');
    e_sorted = sorted(:,1);
    V_sorted = sorted(:,2:end);
    
    for d = 1:d_max
       % choose the d eigenvectors with highest eigenvalues to construct
       % transformation matrix W = [u1 u2 ... ud]       
       W = V_sorted(1:d,:)';
       % project the data on these basis y = W' * images 
       % y with size of d*60000
       y = W' * images;
       % calculate the mean miu_j and covariance sigma_j of digit class j
       % with repect to y?
       miu_j = cell(1,10);% miu_j{j} with dimension d*1
       sigma_j = cell(1,10);% sigma_j{j} with dimension d*d
       X_j = cell(1,10);      
       for p = 1:n
          lab = labels(p) + 1;
          X_j{lab} = [X_j{lab} y(:,p)];
       end
       for j = 1:10
          miu_j{j} = mean(X_j{j}(:,:),2);
          sigma_j{j} = cov(X_j{j}(:,:)');
       end
       %% for testing
       % substract the mean vector of the training data miu      
       % project the testing data on the learned basis y_test = W'*(x_test-miu)
       yt = W' * img_test;
       % calculate the likelihood value of the projected data for each
       % class, using the miu_j and sigma_j learned before
       pxj = zeros(nt,10);
       for j = 1:10
           pxj(:,j) = mvnpdf(yt',(miu_j{j}(:))',sigma_j{j}(:,:));
       end
       [M,I(:,d)] = max(pxj,[],2);   
       % assosiate the input to the class yielding the highest likelihood
       % value
       err_d(d) = 1/nt*sum((I(:,d)-lab_test-ones(nt,1))~= 0)*100;
    end
    % find the err_opt, d_opt    
    [err_opt, d_opt] = min(err_d);
    % find the confusion matrix
    %confusion_mat  = confusionmat(lab_test+ones(nt,1),I(:,d_opt),'order',[1 2 3 4 5 6 7 8 9 10]');
    lab_plus = lab_test + ones(nt,1);
    confusion_mat  = confusionmat(lab_plus,I(:,d_opt),'order',[1 2 3 4 5 6 7 8 9 10]');
    % plot err_d for d = 1:d_max
    clf;
    plot(1:d_max,err_d,'-o','linewidth',2);
    hold on;
    plot(d_opt, err_opt,'rx','MarkerSize',16,'linewidth',2);
    xlabel('d');
    ylabel('Classification Errors  [%]');
    legend({'Classification Errors','Optimal d'},'FontSize',16);
    grid on;
end