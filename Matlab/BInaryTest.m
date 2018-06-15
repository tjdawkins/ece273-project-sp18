%% MNIST DATA SET
% Load teh MNIST Training/Test data and labels (if not loaded)
if ~exist('dataLoadedMNIST','var')
    
    t_images = loadMNISTImages('Data/t10k-images.idx3-ubyte');
    t_labels = loadMNISTLabels('Data/t10k-labels.idx1-ubyte');
    tr_images = loadMNISTImages('Data/train-images.idx3-ubyte');
    tr_labels = loadMNISTLabels('Data/train-labels.idx1-ubyte');
    dataLoadedMNIST = true;

end

%% Defined Classes for comparison and prepare data
% Masks for data selection
c1mask      = tr_labels == 1;
c2mask      = tr_labels == 0; %| tr_labels == 5;
c1mask_test = t_labels == 0;
c2mask_test = t_labels == 7; %| t_labels==5;

% Get observation data selected classes
	% Training
    tr_img_c1 = tr_images(:,c1mask);
    tr_img_c2 = tr_images(:,c2mask);
    
    % Take subset of examples
    r = .02; % Ratio of total examples to use
    tr_img_c1 = tr_img_c1(:,1:round(size(tr_img_c1,2)*r));
    tr_img_c2 = tr_img_c2(:,1:round(size(tr_img_c2,2)*r));
    
    x = [tr_img_c1 tr_img_c2];
    labs      = [ones(size(tr_img_c1,2),1); -1*ones(size(tr_img_c2,2),1)];

    
    % Test
    te_img_c1 = t_images(:,c1mask_test);
    te_img_c2 = t_images(:,c2mask_test);
    x_test = [te_img_c1 te_img_c2];
    test_labs = [ones(size(te_img_c1,2),1); -1*ones(size(te_img_c2,2),1)];



%% Optimization Problem
cvx_begin
    variables B(784) B0
    labs' .* (B'*x + B0) >= 1;    
    minimize(B'*B)
cvx_end


%% Test the classifier on Test Data
g = B'*x_test + B0;
g(g>=0) = 1;
g(g<0) = -1;

% Calculate Error
num_errs = sum(g'~=test_labs)
err_rate = num_errs/length(test_labs)



%% Data Visualization

num_pts_c1 = size(te_img_c1,2);
num_pts_c2 = size(te_img_c2,2);
NUM_VIS_PTS = max(num_pts_c1,num_pts_c2);
%NUM_VIS_PTS = 100;
x1_r2 = zeros(2,NUM_VIS_PTS);
x2_r2 = zeros(2,NUM_VIS_PTS);
% data_cov_1 = cov(te_img_c1');
% data_cov_2 = cov(te_img_c2');
% [data_cov_pc1, ~] = eig(data_cov_1);
% [data_cov_pc2, ~] = eig(data_cov_2);
% data_cov_pc1 = data_cov_pc1(:,end-1:end);
% data_cov_pc2 = data_cov_pc2(:,end-1:end);

% data_cov = cov([te_img_c1 te_img_c2]');
% [data_cov_pc, ~] = eig(data_cov);
% data_cov_pc = data_cov_pc(:,end-1:end);

%%
B_r2 = MapPointsR2(B,B,B0);


for i = 1:NUM_VIS_PTS
    x1_r2(:,i) = MapPointsR2(te_img_c1(:,randi(num_pts_c1)),B, B0);
    x2_r2(:,i) = MapPointsR2(te_img_c2(:,randi(num_pts_c2)),B, B0);
end
%%


x1 = min([x1_r2(1,:) x2_r2(1,:)]):max([x1_r2(2,:) x2_r2(2,:)]);
figure
plot(x1,(B_r2(1)*x1 + B0)/B_r2(2))
hold on
scatter(x1_r2(1,:),x1_r2(2,:),'rx')
scatter(x2_r2(1,:),x2_r2(2,:),'bo')
hold off

%% Soft margin CCCCCC

C=1;

cvx_begin
    variables Bs(784) Bs0 z(length(labs))
    labs' .* (Bs'*x + Bs0) + z' >=1;
    z >= 0;
    minimize(Bs'*Bs + C*sum(z));
cvx_end






