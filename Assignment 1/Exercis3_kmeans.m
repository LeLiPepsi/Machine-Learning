function Exercis3_kmeans(data, init_cluster, k)
    %% initialization: choose k random vectorc as an initial mean set
     X = data;
     [n t d] = size(X);% n*t datas, each data with dimension d    
     Y = init_cluster;
     J_old = 0;
     deltaJ = 1;
     label_vec = zeros(n,t);
     dis_vec = zeros(n,t,k);
     while deltaJ > 10^-6
    %% for each data poin, find the closed and label them
     % the dataset is divided into k classes
        for i=1:n
           for j=1:t
               I = 0;
               M = 0;
               for p=1:k
                    sum = 0;
                    sum = (X(i,j,1)-Y(p,1)).^2 + (X(i,j,2)-Y(p,2)).^2 + (X(i,j,3)-Y(p,3)).^2;
                    % dis_vec(i,j,k) = sqrt(sum((X(i,j,:)-Y(p,:)).^2));
                    dis_vec(i,j,p) = sqrt(sum);
               end
               [M,I] = min(dis_vec(i,j,:));
               label_vec(i,j) = I;
           end
        end     
    %% from the current clusters, their mean vectors are updated
        count = zeros(1,k);
        sum_k = zeros(k,3);
        for i=1:n
           for j=1:t
               get_label = label_vec(i,j);
               sum_k(get_label,:) = sum_k(get_label,:) + [X(i,j,1) X(i,j,2) X(i,j,3)];
               count(get_label) = count(get_label) + 1;
           end
        end
        for q=1:k
           Y(q,:) = sum_k(q,:)/count(q); 
        end
    %% calculate the total distortion, the sum of the distance between
     %  each data point and its closest cluster mean 
     J_new = 0;
     for i=1:n
        for j=1:t
            g_label = label_vec(i,j);
            sum = 0;
            sum = (X(i,j,1)-Y(g_label,1)).^2 + (X(i,j,2)-Y(g_label,2)).^2 + (X(i,j,3)-Y(g_label,3)).^2;
            J_new = J_new + sqrt(sum);
        end
     end
    %% evaluate the convergece
     deltaJ = abs(J_new - J_old);
     J_old = J_new;
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