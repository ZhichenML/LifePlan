function PCA_demo()
load('Prevideo_100.mat');
% n_tr = numel(training(:,1));
% n_te = numel(testing(:,1));
% temp = mapstd([training; testing]')';
% training = temp(1:n_tr,:);
% testing = temp(n_tr+1:end,:);
options.ReducedDim = 0;
[eigvector, eigvalue] = PCA(training, options)%
train = training*eigvector;
test = testing*eigvector;
figure; hold on;
for class_ind = 1:length(unique(training_label))
    scatter3(train(training_label == class_ind,1),train(training_label == class_ind,2),train(training_label == class_ind,3));
end
for class_ind = 1:length(unique(training_label))
    scatter3(test(testing_label == class_ind,1),test(testing_label == class_ind,2),test(testing_label == class_ind,3),'filled');
end
legend('tr1','tr2','te1','te2','C5');
err_points = [1 2 3 4 6 7 9 ];
for i = 1:length(err_points)
    text(test(err_points(i),1),test(err_points(i),2),test(err_points(i),2),num2str(err_points(i)),...
        'FontSize',15);
end
