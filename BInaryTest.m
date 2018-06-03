label_c1=1;
% label_c2=1;
c2mask = tr_labels==3 | tr_labels==5;
tr_img_c1 = tr_images(:,tr_labels==label_c1);
tr_img_c2 = tr_images(:,c2mask);
%mu_0 = mean(tr_img_1,2);
%mu_1 = mean(tr_img_0,2);

x = [tr_img_c1 tr_img_c2];
labs = ones(size(tr_img_c1,2),1);
labs = [labs; -1*ones(size(tr_img_c2,2),1)];

%Cov = pinv(cov(tr_images'));

%B = 2*Cov*(mu_1 - mu_0);
%B0 = mu_0'*Cov*mu_0 - mu_1'*Cov*mu_1;

cvx_begin
    variables B(784) B0
    
    minimize(B'*B)
    labs' .* (B'*x + B0) >= 1
    
cvx_end


%% Test the data
c2mask_test = t_labels==3 | t_labels==5;
te_img_c1 = t_images(:,t_labels==label_c1);
te_img_c2 = t_images(:,c2mask_test);
test_labs = [ones(size(te_img_c1,2),1); zeros(size(te_img_c2,2),1)];

x_test = [te_img_c1 te_img_c2];

g = B'*x_test + B0;
g(g>=0) = 1;
g(g<0) = 0;

num_errs = sum(g'~=test_labs);
err_rate = num_errs/length(test_labs);










