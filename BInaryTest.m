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
c2mask      = tr_labels == 3 | tr_labels == 5;
c1mask_test = t_labels == 1;
c2mask_test = t_labels==3 | t_labels==5;

% Get observation data selected classes
	% Training
    tr_img_c1 = tr_images(:,c1mask);
    tr_img_c2 = tr_images(:,c2mask);
    x = [tr_img_c1 tr_img_c2];
    
    % Test
    te_img_c1 = t_images(:,c1mask_test);
    te_img_c2 = t_images(:,c2mask_test);
    x_test = [te_img_c1 te_img_c2];

% Label data for binary classifier c1 = 1, c2 = -1;
labs      = [ones(size(tr_img_c1,2),1); -1*ones(size(tr_img_c2,2),1)];
test_labs = [ones(size(te_img_c1,2),1); -1*ones(size(te_img_c2,2),1)];

% Optimization Problem
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










