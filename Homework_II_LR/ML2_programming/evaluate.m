targets=csvread('targets.csv');
acc=zeros(1,10);
indices=zeros(1,size(targets,1));
for i=1:10
    fold=csvread(['fold',num2str(i),'.csv']);
    indices(fold(:,1))=indices(fold(:,1))+1;
    acc(i)=sum(targets(fold(:,1),:)==fold(:,2))/size(targets,1);
end
if sum(indices==1)==size(targets,1)
    fprintf('%f\n',mean(acc));
else
    fprintf('not a valid cross-validation\n');
end