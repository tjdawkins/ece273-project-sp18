tr_img_1 = tr_images(:,tr_labels==1);
tr_img_0 = tr_images(:,tr_labels==0);
%mu_0 = mean(tr_img_1,2);
%mu_1 = mean(tr_img_0,2);

x = [tr_img_1 tr_img_0];
labs = ones(size(tr_img_1,2),1);
labs = [labs; -1*ones(size(tr_img_0,2),1)];

%Cov = pinv(cov(tr_images'));

%B = 2*Cov*(mu_1 - mu_0);
%B0 = mu_0'*Cov*mu_0 - mu_1'*Cov*mu_1;

cvx_begin
    variables B(784) B0
    
    minimize(B'*B)
    labs' .* (B'*x + B0) >= 1
    
cvx_end


%% Test the data
te_img_1 = t_images(:,t_labels==1);
te_img_0 = t_images(:,t_labels==0);
test_labs = [ones(size(te_img_1,2),1); zeros(size(te_img_0,2),1)];

x_test = [te_img_1 te_img_0];

g = B'*x_test + B0;
g(g>=0) = 1;
g(g<0) = 0;

num_errs = sum(g'~=test_labs);
err_rate = num_errs/length(test_labs);









