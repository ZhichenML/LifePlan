function adaboost
% author: Zhichen Gong
% name = 'Precoffee';
% load ('parate_accuracy_data(Coffee_re_15).mat');% use the 15th distributiuon
load ('PreIris.mat');
% x_train = pop(15).trans;
% y_train = training_label;
% x_test = pop(15).test_trans;
x_train = train;
y_train = train_label;
x_test = test;
y_test = test_label;

[strong_learner,alpha] = adaboost_train(x_train, y_train, 10);
y_test_predict = adaboost_test(strong_learner,alpha,x_train,y_train,x_test);;




