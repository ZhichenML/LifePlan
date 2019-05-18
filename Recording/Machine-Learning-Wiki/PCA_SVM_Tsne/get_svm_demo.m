% this demo load dataset and perform svm classificaion
% the accuracy of svm, as well as related evaluation for the classifier is
% presented.
clear;
%% first divide the trans into trXD and teXD, then employ CV 
st = cputime;
addpath './libsvm'

% rewrite as your mat file name -> name
% name include training (样本数*属性数),testing,training label（样本数*1）, testing label.
datasetname = 'Prevideo_100';
load([datasetname '.mat']);%%%%%%%%%%%%%%


svm_accuracy = 0;


trans = training;
training_label;

% this is svm parameters to be optimized
kps = [0.000001 0.00001 0.0001 0.001 0.01 0.1 0.5 1:10 20 30 50];
nKps= length(kps);
costs = (-10:1:10);
nCosts = length(costs);

nsv = zeros(nKps,nCosts);
acc = zeros(nKps,nCosts);

% using 5-cross validation to tune the parameter
kFold = 5;
classes = unique(training_label);
nClasses = length(classes);
permcell = cell(nClasses,1);
nSample = cell(nClasses,1);
for i=1:nClasses
    nSample{i} = find(training_label==classes(i));
    len = length(nSample{i});
    permcell{i} = randperm(len);
end

%% divide the training data and CV
fold_ind = ones(nClasses,1);
for k=1:kFold 
        train_ind = [];
        test_ind = [];
        for i=1:nClasses % select from each class randomly for cross validation
            fold_size = floor(length(nSample{i}) ./kFold);
            indexTr = permcell{i}([1:fold_ind(i) - 1 fold_ind(i) + fold_size:end]);% front and back terms
            train_ind = [train_ind ; nSample{i}(indexTr)];
            indexTe =  permcell{i}(fold_ind(i):fold_ind(i) + fold_size - 1); % middle terms
            test_ind  = [test_ind;nSample{i}(indexTe)];
            fold_ind(i) = fold_ind(i)+fold_size;
        end
        trX = trans(train_ind,:); 
        tr_label = training_label(train_ind);
        teX = trans(test_ind,:);
        te_label = training_label(test_ind);
       
%% stadarize the data      
        num1 = size(trX,1); % number of training train observations

        X = [trX;teX]; 
        temp = mapstd(X');
        temp = temp';

        trX = temp(1:num1,:);
        teX = temp(num1+1:end,:);

%% CV
         for j=1:nKps
            kp = kps(j);
             for p=1:nCosts
                C = 10^costs(p);

                 opinion = ['-s 0 -c ' num2str(C) ' -gamma ' num2str(kp) ' -t 2 -q'];
                 model = libsvmtrain(tr_label, trX, opinion);
                 nSV = model.totalSV;
                 [Y0, accuracy,decision_value] = libsvmpredict(te_label,teX, model);
                 acc(j,p) = acc(j,p)+accuracy(1);
                 nsv(j,p) =nsv(j,p)+nSV;
             end
         end
    end
 
acc = acc / kFold;
nsv = nsv / kFold;


bestNsv = 0;
[bestAcc,] = max(acc(:));
acci = find(acc(:) == bestAcc);
if(length(acci)>1)
    tmpNSV = nsv(:);
    [bestNsv,] = min(tmpNSV(acci));
end


indexJ = 0;
indexP = 0;
    for j=1:nKps
        for p=1:nCosts
            if(bestNsv==0)
                if(bestAcc==acc(j,p))
                    indexJ = j;
                    indexP = p;
                end
            else
                if(bestAcc==acc(j,p) && bestNsv==nsv(j,p))
                    indexJ = j;
                    indexP = p;
                end
            end
        end
    end


kp = kps(indexJ);
cost = costs(indexP);
C = 10^cost;

valAcc = bestAcc;

% Train final classifier for testing
trainX = training; 
testX = testing;
training_label; testing_label;
%% std
num1 = size(trainX,1);

X = [trainX;testX];
temp = mapstd(X');
temp = temp';

trainX = temp(1:num1,:);
testX = temp(num1+1:end,:);
%% 
opinion = ['-s 0 -c ' num2str(C) ' -gamma ' num2str(kp) ' -t 2' 'q' '-b 1'];%
model = libsvmtrain(training_label, trainX, opinion);
[Y0, accuracy,decision_value] = libsvmpredict(testing_label,testX, model);%, ['-b 1']);
if binary_classification % ROC can be computed for binary classification.
    ROC = zeros(length(decision_value),2);
    for i = 1:length(decision_value)
        predict_label = decision_value >= decision_value(i);
        predict_label = double(predict_label);
        predict_label(predict_label == 0) = 2;
        performance = Evaluation(testing_label,predict_label);
        ROC(i,1) = performance.tpr;
        ROC(i,2) = performance.fpr;
    end
    [sorted_fpr, ind_fpr] = sort(ROC(:,2)); % draw roc curve
    plot(sorted_fpr,ROC(ind_fpr,1));
end
% [a,b,c,auc]=perfcurve(testing_label,decision_value,2);
performance = Evaluation(testing_label,Y0);
% performance.auc = auc;
testAcc = accuracy(1);
svm_accuracy = testAcc;
% svm_auc = auc; 

time = cputime - st;
% save([datasetname,'performance'],'avm')

