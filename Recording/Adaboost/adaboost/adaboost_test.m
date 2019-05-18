function y_test_predict = adaboost_test(strong_learner, alpha, x_train,y_train,x_test)
% Input:
% strong_learner : a vector of weak learners
% alpha : weight for each weak learner
% x_train : an observation for each row 
% y_train : N * 1 
% x_test : an observation for each row

% Output :
% y_test_predict : a colum vection

N = numel(x_test(:, 1)); % test样本数
T = numel(strong_learner); % number of weak learners
y_test = zeors(N, T); %每列是一个分类器对test的预测标签
for t = 1 : T
    y_test(t) = knn (x_train, y_train,x_test,strong_learner(t));
end
alpha = alpha(:);
y_test_predict = y_test * alpha;
y_test_predict = sign(y_test_predict);

end
