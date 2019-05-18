%%*****************************************
%%preparing the data
load Iris
datasize = size(BloodData,1);
ind = randperm(datasize);
train_ind = ind(ceil(ind*0.7));
test_ind = ind(ceil(ind*0.7):end);
train = BloodData(train_ind,:);
train_label = BloodLabel(train_ind);
test = BloodData(test_ind,:);
test_label = BloodData(test_ind);

%%***************************
%%set parameters
no_dims = 2;
initial_dims = 4;
perplexity = 30;


mappedx = tsne(train,train_label,no_dims,initial_dims,perplexity);

scatter(mappedx(:,1),mappedx(:,2),10,train_label)