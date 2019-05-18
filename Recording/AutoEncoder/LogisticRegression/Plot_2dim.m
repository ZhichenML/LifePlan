function Plot_2dim(train, train_label, theta)

label = [0,1];
W = theta(1:end-1);
W = W(:);
b = theta(end);

max_val = max(train);
min_val = min(train);
x1 = linspace(min_val(1),max_val(1),100)';
x2 = linspace(min_val(2),max_val(2),100)';
[v1,v2]= meshgrid(x1,x2);
grid_val = sigmoid([v1(:) v2(:)]*W+b);

figure(1); hold on;
contour(x1,x2,reshape(grid_val,size(x1,1),size(x2,1)),'ShowText','on');
for i = 1:length(unique(train_label))
    scatter(train(find(train_label==label(i)),1), train(find(train_label==label(i)),2));
end
%surf(x1,x2,reshape(grid_val,size(x1,1),size(x2,1)));

% figure; hold on;
% label = unique(trainY);
% for i =1:length(label)
%     scatter(trainX(find(trainY==label(i)),1),trainX(find(trainY==label(i)),2))
% end

end

function sigm = sigmoid(x)
    sigm = 1./(1+exp(-x));
end
