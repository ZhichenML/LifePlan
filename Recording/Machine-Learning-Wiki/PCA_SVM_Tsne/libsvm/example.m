% [label, instance] = libsvmread('heart_scale');
% model = libsvmtrain(label, instance);
% [predict_label, accuracy, dec_value] = libsvmpredict(label,instance,model);
[label, instance] = libsvmread('heart_scale');

shuffle = randperm(size(label,1))';
division = round(0.7 * size(label,1));
training_index = shuffle(1:division);
testing_index = shuffle(division+1:end);
training_instance = instance(training_index);
training_label = label(training_index);
testing_instance = instance(testing_index);
testing_label = label(testing_index);


model = libsvmtrain(training_label, training_instance);
[predict_label, accuracy, dec_value] = ...
    libsvmpredict(testing_label,testing_instance,model);
