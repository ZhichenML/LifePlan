function performance = Evaluation(true_label, predict_label)
% true_label = [1 1 1 2 2 2 2 2 2];
% predict_label = [2 1 1 1 1 2 2 2 2];


Class = unique(true_label);
nClass = numel(Class);
Class_size = zeros(1,nClass);
for i = 1:nClass
    Class_size(i) = sum(true_label == i);
end

[~, major] = max(Class_size);
[~, minor] = min(Class_size);

disp('minor class:'); disp(Class_size(minor))
disp('major class:'); disp(Class_size(major))

true_positive = sum(true_label == minor & predict_label == minor);
true_negative = sum(true_label == major & predict_label == major);
false_positve = sum(true_label == major & predict_label == minor);
false_negative = sum(true_label == minor & predict_label == major);
tpr = true_positive/(true_positive+false_negative);
fpr = false_positve/(false_positve+true_negative);
tnr = true_negative/(true_negative+false_positve);

 recall = true_positive/(true_positive+false_negative);
 precision= true_positive/(true_positive+false_positve);

accuracy = (true_positive + true_negative)/length(true_label);

performance.true_positive = true_positive;
performance.true_negative = true_negative;
performance.false_positve = false_positve;
performance.false_negative = false_negative;
performance.accuracy = accuracy;
% auc(i,j,l) = (tpr+1-fpr)/2;
                performance.precision = precision;
                performance.recall = recall;
%                 f = (1+beta)*(precision*recall)/(beta*precision+recall);
%                 f = precision;
performance.f_measure = 2*precision*recall/(precision+recall);
%                 fmeasure(i,j,l) = fmeasure(i,j,l)+f;
performance.g_means = sqrt(tpr*tnr);
performance.tpr = tpr;
performance.fpr = fpr;
performance.tnr = tnr;

% for label = 1:nClass
%     true_pos = 
% end
