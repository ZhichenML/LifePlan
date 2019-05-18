clear;clc
addpath ./tSNE_matlab
load('video.mat');
tsne(training,training_label,2,30,30);


% load('PreOSULeaf_PerformanceCRJPareto_lmnn.mat')
% for i = 1:size(trainX_pareto)%1
%     train = trainX_pareto{i}*A_pareto{i};
%     test = testX_pareto{i}*A_pareto{i};
% %     subplot(1,2,1)
% %     tsne(trainl,training_labell,2,20,15)
% %     subplot(1,2,2)
%     tsne(test,testing_label,3,100,20)
% end
% 
% % trainl = [train; repmat(train(find(training_label==6),:)*(1+rand*1),1,1);...
% %     repmat(train(find(training_label==2),:)*(1+rand*1),1,1)];
% % training_labell = [training_label;ones(15,1)*6;ones(sum(training_label==2),1)*2];
% % trainl = [trainl;repmat(train(find(training_label==2),:)*(1+rand*1),1,1)];
% % training_labell=[training_labell; ones(length(training_label==2),1)];