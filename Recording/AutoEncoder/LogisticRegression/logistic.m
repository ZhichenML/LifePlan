% import the data
load banana.mat
x = banana.x;
t = banana.t;
% load overlap.mat;
% x = trainX; t= trainY;
x = bsxfun(@rdivide, bsxfun(@minus,x,mean(x)), (max(x)-min(x)));
t(t==-1) = 0; % sigmoid function ranges in [0,1]

% initialize parameters
Kfold = 10;
theta = Initializepara(numel(x(1,:)));
lambda = 0.0001; % weight decay coefficient

% prepare train and test data
label = unique(t);
nclass = length(label);
sample = cell(nclass,1);
permcell = cell(nclass,1);
for i = 1:nclass
    sample{i} = find(t == label(i));
    nsample(i) = length(sample{i});
    permcell{i} = randperm(nsample(i));
end

fold_ind = ones(nclass,1);
train_ind = cell(Kfold,1);
test_ind = cell(Kfold,1);
for i = 1:Kfold
    for j = 1:nclass
        fold_size = floor(nsample(j)/Kfold);
        train_ind{i} = [train_ind{i}; sample{j}(permcell{j}([1:fold_ind(j)-1 fold_ind(j)+fold_size:end]))];
        test_ind{i} = [test_ind{i}; sample{j}(permcell{j}([fold_ind(j):fold_ind(j)+fold_size-1]))];
        fold_ind(j) = fold_ind(j) + fold_size;
    end
end

% K-fold training
trainerror = nan(Kfold,1);
testerror = nan(Kfold,1);
for i = 1: Kfold
    train = x(train_ind{i},:);
    train_label = t(train_ind{i});
    test = x(test_ind{i},:);
    test_label = t(test_ind{i});
    
    [cost, grad] = calcu_logistic_grad(train, train_label, theta, lambda);
    
    numgrad = Numgrad(@(x)calcu_logistic_grad(train,train_label,x, lambda),theta);
    
    %============================================================================
    % optimize by minfunc
    %  Randomly initialize the parameters
    theta = Initializepara(size(x,2));
    
    %  Use minFunc to minimize the function
    addpath minFunc/
    %options.Method = 'newton0';
    options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
    % function. Generally, for minFunc to work, you
    % need a function pointer with two outputs: the
    % function value and the gradient. In our problem,
    % sparseAutoencoderCost.m satisfies this.
    options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run
    options.display = 'on';
    
    
    [opttheta, cost] = minFunc( @(p)calcu_logistic_grad(train, train_label, p, lambda), theta, options);
%    Plot_2dim(train,train_label,opttheta);
    W = opttheta(1:end-1); W = W(:); b = opttheta(end);
    predicttrain = round(1./(1+exp(train*W+b)));
    predicttest = round(1./(1+exp(test*W+b)));
    
    trainerror(i) = sum(predicttrain~=train_label)/numel(train_label);
    testerror(i) = sum(predicttest~=test_label)/numel(test_label);
end

fprintf('Mean train error is %f%% \n', mean(trainerror)*100)
fprintf('Mean test error is %f%% \n', mean(testerror)*100)