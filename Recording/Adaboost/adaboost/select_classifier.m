function [best_k,J, label_predict] = select_classifier(x_train, y_train,weight)
% search for a classifier minimizing the weighted error function
% 用留一法选择ｋｎｎ　accuracy 最高的ｋ,并返回weighted error和 对应的预测标签
%Input:
% x_train,y_train,row-wise 
% weight : a column vector
%Output:
% label_predict : a column vector
N = numel(x_train(:, 1)); % 样本数
weighted_error = zeros(N,1); %每个Ｋ值对应的分类器错误分类的样本数的权值和
y_predict = zeros(N, N); %每一行对应一个对x_train的预测标签
for k = 1 : 2 %grid search for k
    for j = 1 : N  % for each observation
        index_train = [1:j-1 j+1:N]; %去除第ｊ个样本
        y_predict(k, j) = knn(x_train(index_train,:),y_train(index_train,:),x_train(j,:),k);
    end
    weighted_error(k) = sum(weight(find(y_predict(k, :) ~= y_train'))); % 记录错误分类的样本数的权值
end

[J, best_k] = min(weighted_error); % search for minimizing the weighted error function
label_predict = y_predict(best_k, :)';