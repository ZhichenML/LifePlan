function final_accuracy = knnmain(pop,name)
%% ����name���ݼ����շ���׼ȷ��

%% ����ÿ��ת��������÷�����
load(strcat('./dataset/',strcat(name,'.mat')));
train_label = training_label_train;
validation_label = training_label_validation;
% load(strcat('./out/',strcat(name,'_validation_performance.mat')));
npop = numel(pop);
pop_result = zeros(npop,2);  

 
for i = 1:npop
    training = pop(i).trans;  validation =pop(i).validation_trans;   % ����ÿ����������������ʹ������
    class_accuracy = zeros(1,length(train_label)); % �������п��ܵ� k (for knn)
    for k = 1:length(train_label)
        label = knn(training,train_label,validation,k);
        class_accuracy(k) = sum(validation_label == label)/length(validation_label);
    end
%     sprintf('%d finished.',i);
    [pop_result(i,1),pop_result(i,2)] = max(class_accuracy); %��i��ת��������õķ�������¼(accuracy,k for knn)
end
% [v, ind] = max(result(:,1));
% disp(sprintf('bestperformance: %f  k= %d person=%d',1-result(ind,1),result(ind,2),ind))

%% ������ͶƱ
[v, ind] = sort(pop_result(:,1), 'descend'); % ind: �����Դﵽ�ķ���Ч����������ת����,ind(i)��ʾ����Ч����i�õķ��������
final_accuracy = zeros(1,npop); %��������ͶƱ��ת������:1~npop
for m=1:npop
%     m = round(0.09 * npop); % ȡ��ת��������9% ͶƱ
    m_vote = zeros(length(validation_label),m); %���������*m, m��������ͶƱ�������ս��,ÿ�ж�Ӧһ�������� 
    
    for i = 1:m
        training = pop(ind(i)).trans; validation = pop(ind(i)).validation_trans;
        m_vote(:,i) = knn(training,train_label,validation,pop_result(ind(i),2)); % knn�Ե�i��������������ͶƱ
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
%         xlabel('trainMSE(�������)');ylabel('true classe probability(��������)');
%         title('Coffee knn ��Ⱥ���������ֲ�');
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