function [strong_learner, alpha] = adaboost_train(x_train, y_train, T)
% Input : 
% x_train : 样本数*维数 
% y_train : 样本数*1        
N = numel(x_train(:,1)); % number of training examples
weight = ones(N, 1)/N; % initialize weight for each example
alpha = zeros(T, 1); %initialize weight for each weak classifier
strong_learner = zeros(T, 1); % initialize k for t weak classifier (knn)

for t = 1 : T
    %train weak learner
    [learner,J, y_predict] = select_classifier(x_train,y_train,weight);
    
    %compute error
    epsilon = J /sum(weight);
    
    if (epsilon >=0.5)
        T = T - 1;
        break;
    end
    
    strong_learner(t) = learner;
    
    %compute alpha
    alpha(t) =  log((1-epsilon)/epsilon);
    
    %update data weight
    weight =  weight.*exp(alpha*abs(y_predict-y_train)/2);
end
    
    
