function data_reallocation()
% train-test ratio : 0.7 0.3
load('video.mat');
label = cell(2,1);
label{1} = find(training_label == 1);
label{2} = find(training_label ==0 );

