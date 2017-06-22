function Exercise3_nubs(data,k)
    %% initialization: calculate a center for all data points
    X = data;
    [n t d] = size(X);
    Y = zeros(k,3);
    rvec = [0.08 0.05 0.02]';
    % for initialization, all data points are assigned with class label 1
    label_vec = ones(n,t);
%     for i=1:n
%        for j=1:t
%           Y(1,:) = Y(1,:)+[X(i,j,1) X(i,j,2) X(i,j,3)]/(n*t); 
%        end
%     end
    for i=1:3
       Y(1,i) = mean(mean(X(:,:,i))); 
    end
    J = zeros(1,k);
   %% choose a class which has the largest distortion among current classes
    for K=1:k   
    I = 0;
    % compute distortion functions J(K)
    sum = zeros(1,K);
    for i = 1:n
       for j = 1:t
           lab = label_vec(i,j);
           sum(lab) = (X(i,j,1)-Y(lab,1))^2 + (X(i,j,2)-Y(lab,2))^2 + (X(i,j,3)-Y(lab,3))^2;
           J(lab) = J(lab) + sqrt(sum(lab));
       end
    end
    % the index of the worst class will be saved in I
    [Max, I] = max(J);      
    %% split the class into two subclasses by using a small random vector
    % temporary data group Y_t with size 2x3 storing y+v and y-v
    % respectively
    Y_t = zeros(2,3);
    Y_t(1,:) = Y(I,:) + rvec';
    Y_t(2,:) = Y(I,:) - rvec';
    for i = 1:n    
        for j = 1:t
            dis_vec = zeros(1,2);
            if label_vec(i,j) == I
                dis_vec(1) = norm([X(i,j,1) X(i,j,2) X(i,j,3)]-Y_t(1,:));
                dis_vec(2) = norm([X(i,j,1) X(i,j,2) X(i,j,3)]-Y_t(2,:));
                if dis_vec(1) >= dis_vec(2)
                    ;
                else
                    label_vec(i,j) = K + 1;
                end
            else
                ;
            end
        end
    end 
    %% update the code vectors
    countI = 0;
    countK = 0;
    Y(I,:) = zeros(1,3);
    Y(K+1,:) = zeros(1,3);
    for i = 1:n
       for j = 1:t
          if label_vec(i,j) == I
              Y(I,:) = Y(I,:) + [X(i,j,1) X(i,j,2) X(i,j,3)];
              countI = countI + 1;
          elseif label_vec(i,j) == K+1
              Y(K+1,:) = Y(K+1,:) + [X(i,j,1) X(i,j,2) X(i,j,3)];
              countK = countK + 1;
          end
       end
    end
    Y(I,:) = Y(I,:) / countI;
    Y(K+1,:) = Y(K+1,:) / countK;
    end
    %% figures
     clf;
     for i=1:n
        for j=1:t
           g_label = 0;
           g_label = label_vec(i,j);
           switch g_label
               case 1
                   scatter3(X(i,j,1),X(i,j,2),X(i,j,3),'filled','blue');
               case 2
                   scatter3(X(i,j,1),X(i,j,2),X(i,j,3),'filled','black');
               case 3
                   scatter3(X(i,j,1),X(i,j,2),X(i,j,3),'filled','red');
               case 4
                   scatter3(X(i,j,1),X(i,j,2),X(i,j,3),'filled','green');
               case 5
                   scatter3(X(i,j,1),X(i,j,2),X(i,j,3),'filled','magenta');
               case 6
                   scatter3(X(i,j,1),X(i,j,2),X(i,j,3),'filled','yellow');
               case 7
                   scatter3(X(i,j,1),X(i,j,2),X(i,j,3),'filled','cyan');
           end
           hold on;
        end
     end
end