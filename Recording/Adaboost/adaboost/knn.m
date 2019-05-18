function test_label = knn(training,train_label,test,k)
%%
% train : N*P,each row represents an observation
% train_label : N*1 label for each row in trian
% test : M*P,each row represents an boservation to classify
% k : number of neighbors
%%

test_label = zeros(size(test, 1),1); % label for each row in test

for i = 1:size(test, 1)
    neighbors = get_neighbors(training, test(i,:), k); %find k neighbors for ith observation from train
    test_label(i) = votes (neighbors,train_label);
end

end


% training =[1 1;1 1.2;1.1 1.1;2 2.1;2 2;2.1 2.3];
% train_label = [1;1;1;2;2;2];
% test=[2.4 2.2; 0.9 1.2];
% knn(training,train_label,test,3)
% scatter(training(:,1),training(:,2),'filled')
% hold on
% scatter(test(:,1),test(:,2),'filled')

