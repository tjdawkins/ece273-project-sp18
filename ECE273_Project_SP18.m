%% Data Import / Generate

% Load the MNIST Training/Test data and labels (if not loaded)
if ~exist('dataLoadedMNIST','var')
    
    t_images = loadMNISTImages('Data/t10k-images.idx3-ubyte');
    t_labels = loadMNISTLabels('Data/t10k-labels.idx1-ubyte');
    tr_images = loadMNISTImages('Data/train-images.idx3-ubyte');
    tr_labels = loadMNISTLabels('Data/train-labels.idx1-ubyte');
    dataLoadedMNIST = true;

end

%% Generate Data
% Variable naming : {data}{class}{structure}
% Data: x = examples / y = labels
% Class: 1 = class 1 / 2 = class 2
% Structure: ls = linearly separable /  lns = linear non-separable / r = radial
%% Generate Linearly Separable Data
ndata = 500;
ntest = 1000;
s = 2;
r = 45;
o = 3.5 * s;
L = 40;
x1ls = gen_data_linear_r2(ndata,L,s,r,[0;+o]);
x2ls = gen_data_linear_r2(ndata,L,s,r,[0;-o]);
y1ls = ones(size(x1ls,2),1);
y2ls = -ones(size(x2ls,2),1);

% Test Data Linearly Seperable
x1lst = gen_data_linear_r2(ntest,L,s,r,[0;+o]);
x2lst = gen_data_linear_r2(ntest,L,s,r,[0;-o]);
y1lst = ones(size(x1ls,2),1);
y2lst = -ones(size(x2ls,2),1);
xlst = [x1lst x2lst];
ylst = [y1lst y2lst];

figure
plot(x1ls(1,:),x1ls(2,:),'rx')
hold on
plot(x2ls(1,:),x2ls(2,:),'bx')

%% Generate Linearly Separablish Data
ndata = 500;
s = 2;
r = 45;
o = 2 * s;
L = 40;
x1lns = gen_data_linear_r2(ndata,L,s,r,[0;+o]);
x2lns = gen_data_linear_r2(ndata,L,s,r,[0;-o]);
y1lns = ones(size(x1lns,2),1);
y2lns = -ones(size(x2lns,2),1);

% Test Data Linearly Non Seperable
x1lnst = gen_data_linear_r2(ntest,L,s,r,[0;+o]);
x2lnst = gen_data_linear_r2(ntest,L,s,r,[0;-o]);
y1lnst = ones(size(x1lns,2),1);
y2lnst = -ones(size(x2lns,2),1);
xlnst = [x1lnst x2lnst];
ylnst = [y1lnst y2lnst];



figure
plot(x1lns(1,:),x1lns(2,:),'rx')
hold on
plot(x2lns(1,:),x2lns(2,:),'bx')

%% Generate some RBF type Data
u = [5;5];
x1r = gen_data_radial_r2(1000,u,4,12);
x2r = gen_data_radial_r2(500,u,.75);
y1r = ones(size(x1r,2),1);
y2r = -ones(size(x2r,2),1);

% Test Data Radial
x1rt = gen_data_linear_r2(ntest,L,s,r,[0;+o]);
x2rt = gen_data_linear_r2(ntest,L,s,r,[0;-o]);
y1rt = ones(size(x1rt,2),1);
y2rt = -ones(size(x2rt,2),1);
xrt = [x1rt x2rt];
yrt = [y1rt y2rt];


figure
plot(x1r(1,:),x1r(2,:),'ro')
hold on
plot(x2r(1,:),x2r(2,:),'bo')

%% Train Models on Gen Data
% Training Data
% Data: x = examples / y = labels
% Class: 1 = class 1 / 2 = class 2
% Structure: ls = linearly separable /  lns = linear non-separable / r = radial
xls = [ x1ls x2ls];
yls = [ y1ls; y2ls];
xlns = [ x1lns x2lns];
ylns = [ y1lns; y2lns];
xr = [ x1r x2r];
yr = [ y1r; y2r];

%% Train Models on Datasets
cvx_precision high
cvx_solver sedumi
c = 1;
% Train Dual and Primal on Seperable and Non-Separable Data
% Seperable
[Bdls, B0dls, adls, SVdls, ysdls] = svm_dual(xls,yls,0);
[Bpls, B0pls, SVpls, yspls] = svm_primal(xls,yls,0);
% Non-Seperable
[Bdlns, B0dlns, adlns, SVdlns, ysdlns, zdlns] = svm_dual(xlns,ylns,c);
[Bplns, B0plns, SVplns, ysplns, zplns] = svm_primal(xlns,ylns,c);

% Train Dual on Radial Data with rbf kernel
[Br, B0r, ar, SVr, ysr, zr] = svm_dual(xr,yr,0,'rbf',.25);

%% Visualie Hard Margin / Seperable
figure
visualize_svm_linear(x1ls, x2ls, Bpls, B0pls, SVpls, 'Linearly Seperable - Primal')
figure
visualize_svm_linear(x1ls, x2ls, Bdls, B0dls, SVdls, 'Linearly Seperable - Dual')
figure
visualize_svm_linear(x1lns, x2lns, Bplns, B0plns, SVplns, 'Linearly Seperable - Primal',zplns)
figure
visualize_svm_linear(x1lns, x2lns, Bdlns, B0dlns, SVdlns, 'Linearly Non Seperable - Dual',zdlns)


%% Classify New Expamples




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






