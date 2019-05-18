function [cost, grad] = calcu_logistic_grad(train, train_label, theta, lambda)
% reconize parameters
W = theta(1:end-1);
b = theta(end);

% calculate cost and grad
%cost = sum(train_label.*log(sigmoid(train*W+b*ones(size(train,1),1))) + ...
%    (1- train_label).*(log(1-sigmoid(train*W+b*ones(size(train,1),1)))));
square_ave = sum(train_label.*(train*W+b*ones(size(train,1),1) ) +log(1-sigmoid(train*W+b*ones(size(train,1),1))) )/size(train,1);
weight_decay = lambda/2*sum(sum(W.*W));
cost = square_ave + weight_decay;

Wgrad = ((train_label - sigmoid(train*W+b))' * train )' /size(train,1)+ lambda*W;
bgrad = sum(train_label - sigmoid(train*W+b))/size(train,1);

grad = [Wgrad(:); bgrad];
end

function sigm = sigmoid(x)
    sigm = 1./(1+exp(-x));
end
