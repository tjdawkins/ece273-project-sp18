% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
tr_images = loadMNISTImages('Data/t10k-images.idx3-ubyte');
tr_labels = loadMNISTLabels('Data/train-labels.idx1-ubyte');
 