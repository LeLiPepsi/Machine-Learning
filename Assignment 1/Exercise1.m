function par = Exercise1(k)
    % input k - cross validation parameter k
    % output par - cell array of size 1x3, each cell par{i} contaions the
    % learnend parameters values
    par = cell([1 3]);
    p1_m = 6;
    p2_m = 6;
    load('Data.mat');
    n = length(Input);
    %% generate k subsamples
    sub_s = cell([2 k]);
    sub_tr = cell([2 k]);
    % testing data sub_s{1/2,i} for subsample i
    % training data sub_tr{1/2,i} for subsample i
    for K = 1:k
        sub_s{1,K} = Input(:,1+(K-1)*n/k:K*n/k)';
        sub_s{2,K} = Output(:,1+(K-1)*n/k:K*n/k)';
        if K==1
            sub_tr{1,K} = Input(:,1+K*n/k:end)';
            sub_tr{2,K} = Output(:,1+K*n/k:end)';
        elseif K==k
            sub_tr{1,K} = Input(:,1:(K-1)*n/k)';
            sub_tr{2,K} = Output(:,1:(K-1)*n/k)';
        else
            sub_tr{1,K} = [Input(:,1:(K-1)*n/k)' ; Input(:,1+K*n/k:end)'];
            sub_tr{2,K} = [Output(:,1:(K-1)*n/k)' ; Output(:,1+K*n/k:end)'];
        end
    end
    %% for p1 = 1 -> 6, estimate parameters in the equations of x and y.
    % parameters saved in cell array a_x, each cell a_x{i} and a_y{i} contains learned
    % parameters for p1 = i.
    pos_err = zeros(1,p1_m);  
    for p1 = 1:p1_m
        pos_errK = zeros(1,k);
        for K = 1:k
            % training data sub_tr{1/2,K} for subsample K
            % construct Y = X * Beta of multiple linear regression model 
            Y = [];
            Y = sub_tr{2,K}(:,1:2);
            X = ones(length(Y),1);
            for j=1:p1
               X = [X sub_tr{1,K}(:,1).^j sub_tr{1,K}(:,2).^j (sub_tr{1,K}(:,1).*sub_tr{1,K}(:,2)).^j];
            end
            beta_xK = (X'*X)\(X'*Y(:,1));
            beta_yK = (X'*X)\(X'*Y(:,2));
            % testing data sub_s{1/2,K} for subsample K, result is saved in pos_errK(K)
            A = ones(length(sub_s{1,K}),1);
            for j=1:p1
               A = [A sub_s{1,K}(:,1).^j sub_s{1,K}(:,2).^j (sub_s{1,K}(:,1).*sub_s{1,K}(:,2)).^j] ;
            end
            x_ref_K = A * beta_xK;
            y_ref_K = A * beta_yK;
            pos_errK(K) = 1/(n/k)*sum(((sub_s{2,K}(:,1)-x_ref_K).^2+(sub_s{2,K}(:,2)-y_ref_K).^2).^0.5);
        end
        % take average of pos_errK to get overall error pos_err(p1)
        pos_err(p1) = mean(pos_errK);
    end
    %% for p2 = 1 -> 6, estimate parameters in the equation of theta
    % parameters saved in cell array a_y, each cell a_t{j} contains learned
    % parameters for p2 = j.
    ori_err = zeros(1,p2_m);
    for p2=1:p2_m
        ori_errK = zeros(1,K);
        for K = 1:k
           Y = [];
           Y = sub_tr{2,K}(:,3);
           X = ones(length(Y),1);
           for j=1:p2
               X = [X sub_tr{1,K}(:,1).^j sub_tr{1,K}(:,2).^j (sub_tr{1,K}(:,1).*sub_tr{1,K}(:,2)).^j];
           end
           beta_thK = (X'*X)\(X'*Y);
           A = ones(length(sub_s{1,K}),1);
           for j=1:p2
                A = [A sub_s{1,K}(:,1).^j sub_s{1,K}(:,2).^j (sub_s{1,K}(:,1).*sub_s{1,K}(:,2)).^j] ;
           end
           theta_ref_K = A * beta_thK;
           ori_errK(K) = 1/(n/k)*sum(((sub_s{2,K}(:,3)-theta_ref_K).^2).^0.5);           
        end
        ori_err(p2) = mean(ori_errK);
    end
    %% compare position & orientation error to find the optimum of p1_opt&p2_opt
    [min_pos_err, p1_opt] = min(pos_err);
    [min_ori_err, p2_opt] = min(ori_err);    
    %% re-estimate parameters
    % X_pos, X_ori
    X_pos = ones(n,1);
    X_ori = ones(n,1);
    for i=1:p1_opt
        X_pos = [X_pos (Input(1,:)').^i (Input(2,:)').^i ((Input(1,:).*Input(2,:))').^i];
    end
    for j=1:p2_opt
        X_ori = [X_ori (Input(1,:)').^j (Input(2,:)').^j ((Input(1,:).*Input(2,:))').^j];
    end
    par{1} = (X_pos'*X_pos)\(X_pos'*Output(1,:)');
    par{2} = (X_pos'*X_pos)\(X_pos'*Output(2,:)');
    par{3} = (X_ori'*X_ori)\(X_ori'*Output(3,:)');
    
end