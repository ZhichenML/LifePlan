function main
% to find the sub-objective value of the minimum mono-objective, 
% the implementation can be performed to take care of the vector form, that
% is, to turn the sub and mono-objective values into vectors.
x1_set = 0:0.01:1;
x2_set = 0:0.01:1;
f1_set = nan(length(x1_set),length(x2_set));
f2_set = nan(length(x1_set),length(x2_set));
for x1_ind = 1:length(x1_set)
    for x2_ind = 1:length(x2_set)
        x1 = x1_set(x1_ind);
        x2 = x2_set(x2_ind);
        f1_set(x1_ind,x2_ind) = f1(x1,x2);
        f2_set(x1_ind,x2_ind) = f2(x1,x2);
    end
end

%===========linear search===============
weight = 0:0.01:1;
% f1_obj = nan(1,length(weight));
% f2_obj = nan(1,length(weight));
f1_obj = [];
f2_obj = [];
for weight_ind = 1:length(weight)
    objective = f1_set*weight(weight_ind)+f2_set*(1-weight(weight_ind));
    indx = find(objective(:)==min(objective(:)));
    f1_obj = [f1_obj; f1_set(indx)];
    f2_obj = [f2_obj; f2_set(indx)];
% %old implementation    
% %     [min_val,min_ind] = min(objective(:));%min_ind
% % find the minimum objective value under the current weight.
%     [row, col] = ind2sub(size(objective),find(objective==min(objective(:))));
% %     f1_obj(weight_ind) = f1_set(row,col);
% %     f2_obj(weight_ind) = f2_set(row,col);
%     new_row = []; new_col = []; obj = [];
%     for i =1:length(row)
%         obj = [obj;objective(row(i),col(i))];
%         if(objective(row(i),col(i)) == 0)
%             new_row = [new_row row(i)];
%             new_col = [new_col col(i)];
%         end
%     end
%     temp1 = []; temp2 =[];
%     for i = 1:length(new_row)
%         temp1 = [temp1;f1_set(new_row(i),new_col(i))];
%         temp2 = [temp2;f2_set(new_row(i),new_col(i))];
%     end
%     f1_obj = [f1_obj;temp1(:)];
%     f2_obj = [f2_obj;temp2(:)];
% % old implementation
end
%     subplot(1,3,1); imagesc(f1_set);title('f1');subplot(1,3,2);imagesc(f2_set);title('f2');
%     subplot(1,3,3)
    scatter(f1_obj,f2_obj,100,'filled'); hold on;
    scatter(f1_set(:),f2_set(:))
    xlabel('f1');ylabel('f2');
    % find the pareto front using this logical Pareto Front
    front = paretofront([f1_set(:),f2_set(:)]);
    ind = find(front == 1);
    scatter(f1_set(ind),f2_set(ind),100,'filled')
end

function [f1_value] = f1(x1,x2)
f1_value = 4*x1;
end

function [g_value] = g(x2)
g_value = 4-3*exp(-((x2-0.2)/0.02).^2);
end

function [h_value] = h(x1,x2)
if x1<x2
    h_value = 1-(x1/x2)^4;
else
    h_value = 0;
end
end

function [f2_value] = f2(x1,x2)
f2_value = g(x2)*h(f1(x1,x2),g(x2));
end


