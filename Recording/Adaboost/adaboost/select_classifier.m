function [best_k,J, label_predict] = select_classifier(x_train, y_train,weight)
% search for a classifier minimizing the weighted error function
% ����һ��ѡ����accuracy ��ߵģ�,������weighted error�� ��Ӧ��Ԥ���ǩ
%Input:
% x_train,y_train,row-wise 
% weight : a column vector
%Output:
% label_predict : a column vector
N = numel(x_train(:, 1)); % ������
weighted_error = zeros(N,1); %ÿ����ֵ��Ӧ�ķ���������������������Ȩֵ��
y_predict = zeros(N, N); %ÿһ�ж�Ӧһ����x_train��Ԥ���ǩ
for k = 1 : 2 %grid search for k
    for j = 1 : N  % for each observation
        index_train = [1:j-1 j+1:N]; %ȥ���ڣ������
        y_predict(k, j) = knn(x_train(index_train,:),y_train(index_train,:),x_train(j,:),k);
    end
    weighted_error(k) = sum(weight(find(y_predict(k, :) ~= y_train'))); % ��¼����������������Ȩֵ
end

[J, best_k] = min(weighted_error); % search for minimizing the weighted error function
label_predict = y_predict(best_k, :)';