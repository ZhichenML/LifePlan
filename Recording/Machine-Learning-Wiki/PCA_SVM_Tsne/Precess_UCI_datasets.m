% function Precess_UCI_datasets()
% fid = fopen('Adult.data');
% tline = fgetl(fid);
% strrep(tline,'State-gov','')
% % while(tline)
%     
% % end

names = importdata('AdultNames.data');
for i= 1:length(names)
    name_features = regexp(names{i},'\S','split');% ������ʽ��splitģʽ��
    features = regexp(name_features{2},',','split');
end
