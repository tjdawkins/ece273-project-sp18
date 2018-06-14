%% Data Import / Generate

% Load the MNIST Training/Test data and labels (if not loaded)
if ~exist('dataLoadedMNIST','var')
    
    t_images = loadMNISTImages('Data/t10k-images.idx3-ubyte');
    t_labels = loadMNISTLabels('Data/t10k-labels.idx1-ubyte');
    tr_images = loadMNISTImages('Data/train-images.idx3-ubyte');
    tr_labels = loadMNISTLabels('Data/train-labels.idx1-ubyte');
    dataLoadedMNIST = true;

end

% Generate Data

%% Primal / Dual - Hard / Sorft Margins


%% Dual with Kernels

%% Soft Margin

%% SVM with MNIST Data Set: Data and labels 

% Defined Classes for comparison and prepare data
% Masks for data selection
c1mask      = tr_labels == 1;
c2mask      = tr_labels == 0; %| tr_labels == 5;
c1mask_test = t_labels == 1;
c2mask_test = t_labels == 0; %| t_labels==5;

% Get observation data selected classes
	% Training
    tr_img_c1 = tr_images(:,c1mask);
    tr_img_c2 = tr_images(:,c2mask);
    
    % Take subset of examples
    r = .02; % Ratio of total examples to use
    tr_img_c1 = tr_img_c1(:,1:round(size(tr_img_c1,2)*r));
    tr_img_c2 = tr_img_c2(:,1:round(size(tr_img_c2,2)*r));
    
    x = [tr_img_c1 tr_img_c2];
    y = [ones(size(tr_img_c1,2),1); -1*ones(size(tr_img_c2,2),1)];

    
    % Test
    te_img_c1 = t_images(:,c1mask_test);
    te_img_c2 = t_images(:,c2mask_test);
    x_test = [te_img_c1 te_img_c2];
    y_test = [ones(size(te_img_c1,2),1); -1*ones(size(te_img_c2,2),1)];

%% SVM with MNIST Data Set: Models

% Train Model
% Primal Hard
[Bph, B0ph, SVph, ysph, ~] = svm_primal(x,y,0);
% Dual Hard
[Bdh, B0dh, asdh, SVdh, ysdh] = svm_dual(x,y,0);

% Primal Soft
[Bps, B0ps, SVps, ysps, ~] = svm_primal(x,y,1);
% Dual Soft
[Bds, B0ds, asds, SVds, ysds] = svm_dual(x,y,1);

% Dual Kernel rbf
[Bdhk, B0dhk, asdhk, SVdhk, ysdhk] = svm_dual(x,y,0,'rbf',.25);

%% Testing
N=length(y_test);

yph = decision_primal(Bph, B0ph, x_test);
er_ph = sum(yph~=y_test)/N;
ydh = decision_dual(asdh, ysdh, B0dh, SVdh, x_test);
er_dh = sum(ydh~=y_test)/N;

yps = decision_primal(Bps, B0ps, x_test);
er_ps = sum(yps~=y_test)/N;
yds = decision_dual(asds, ysds, B0ds, SVds, x_test);
er_ds = sum(yds~=y_test)/N;
%
ydhk= decision_dual(asdhk, ysdhk, B0dhk, SVdhk,x_test,'rbf',0.25);
er_dhk = sum(ydhk~=y_test)/N;






