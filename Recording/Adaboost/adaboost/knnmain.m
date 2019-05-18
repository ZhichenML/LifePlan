function final_accuracy = knnmain(pop,name)
%% 返回name数据集最终分类准确度

%% 计算每个转换器的最好分类结果
load(strcat('./dataset/',strcat(name,'.mat')));
train_label = training_label_train;
validation_label = training_label_validation;
% load(strcat('./out/',strcat(name,'_validation_performance.mat')));
npop = numel(pop);
pop_result = zeros(npop,2);  

 
for i = 1:npop
    training = pop(i).trans;  validation =pop(i).validation_trans;   % 读入每个分类器的样本点和待分类点
    class_accuracy = zeros(1,length(train_label)); % 遍历所有可能的 k (for knn)
    for k = 1:length(train_label)
        label = knn(training,train_label,validation,k);
        class_accuracy(k) = sum(validation_label == label)/length(validation_label);
    end
%     sprintf('%d finished.',i);
    [pop_result(i,1),pop_result(i,2)] = max(class_accuracy); %第i个转换器的最好的分类结果记录(accuracy,k for knn)
end
% [v, ind] = max(result(:,1));
% disp(sprintf('bestperformance: %f  k= %d person=%d',1-result(ind,1),result(ind,2),ind))

%% 分类结果投票
[v, ind] = sort(pop_result(:,1), 'descend'); % ind: 按可以达到的分类效果降序排列转换器,ind(i)表示分类效果第i好的分类器编号
final_accuracy = zeros(1,npop); %遍历参与投票的转换器数:1~npop
for m=1:npop
%     m = round(0.09 * npop); % 取总转换器数的9% 投票
    m_vote = zeros(length(validation_label),m); %待分类点数*m, m个分类结果投票产生最终结果,每列对应一个分类结果 
    
    for i = 1:m
        training = pop(ind(i)).trans; validation = pop(ind(i)).validation_trans;
        m_vote(:,i) = knn(training,train_label,validation,pop_result(ind(i),2)); % knn对第i个分类器产生的投票
    end
    final_label = mode(m_vote,2);
    final_accuracy(m) = sum(final_label == validation_label)/length(validation_label);
end


%% draw
% clear
% clc
% load('result1');load('PreOliveOil_100pop_300g','test_classAbility','F');load('PreOliveOildistribution1');
%     nf=numel(F);
%     c=GetCosts(pop);
%     
%     h=linspace(0,2/3,nf);
%     
%     costs=cell(1,nf);
%     legends=cell(1,nf);
%     
%     for f=1:nf
%         costs{f}=c(:,F{f});
%         legends{f}=['Front ' num2str(f)];
%         
%         color=hsv2rgb([h(f) 1 1]);
%         result1=result';
%         scatter(c(1,:),c(2,:),2.5,result1(1,:),'filled');colorbar horizon;
%         xlabel('trainMSE(拟合能力)');ylabel('true classe probability(分类能力)');
%         title('Coffee knn 种群分类能力分布');
%         hold on;
%         
%     end
%     
%     legend(legends);
%     legend('Location','NorthEastOutside');
%     
%     midterm = mean (result(:,1))+0.0;
%     ind = find (result(:,1)>midterm);
%     for i = 1 : numel(F{1})
%         if any(F{1}(i)==ind)
%         text(pop(F{1}(i)).Cost(1)+10^(-5),pop(F{1}(i)).Cost(2),num2str([F{1}(i),test_classAbility(F{1}(i)),result(i)]));%,test_class_accuracy(F{1}(i))
%         end
%     end
%     hold off;