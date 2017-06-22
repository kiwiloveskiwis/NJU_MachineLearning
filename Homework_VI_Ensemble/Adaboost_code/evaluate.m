targets=csvread('targets.csv');
base_list=[1,5,10,100];

for k=1:length(base_list)
    base_num=base_list(k);
    acc=zeros(1,10);
    for i=1:10
        fold=csvread(sprintf('experiments/base%d_fold%d.csv',base_num,i));
        acc(i)=sum(targets(fold(:,1),:)==fold(:,2))/size(fold,1);
    end
    fprintf('%f\n',mean(acc));

end