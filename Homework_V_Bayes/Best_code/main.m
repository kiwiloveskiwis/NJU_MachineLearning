% input
load('all_data.mat');
X=full(train_data);
y=train_targets;
Xt=full(test_data);

% parameters
alpha=1;

% compute probability
num_instance=size(X,1);
prior=zeros(5,1);
p_0=zeros(5,2500);
p_1=zeros(5,2500);
mu=zeros(5,2500);
sigma=zeros(5,2500);
for i=0:4
    prior(i+1)=sum(y(:)==i);
    XP=X(y(:)==i,:);
    p_0(i+1,:)=(sum(XP(:,1:2500)==0)+alpha)./(size(XP,1)+2*alpha);
    p_1(i+1,:)=(sum(XP(:,1:2500)==1)+alpha)./(size(XP,1)+2*alpha);
    mu(i+1,:)=mean(XP(:,2501:5000));
    sigma(i+1,:)=std(XP(:,2501:5000));
end

% numeric issues
prior=log(prior);
p_0=log(p_0);
p_1=log(p_1);
max_sigma=max(sigma(:));
sigma=sigma+max_sigma.*0.001;

% testing
num_test=size(Xt,1);
pred=zeros(num_test,1);
for i=1:num_test
    p=prior;
    c=Xt(i,1:2500)';
    p=p+p_1*c+p_0*(1-c);
    c=Xt(i,2501:5000)';
    p=p+sum(-(ones(5,1)*c'-mu).^2./2./(sigma.^2)-log(sigma),2);
    [~, index]=max(p);
    pred(i)=index-1;
end

csvwrite('test_predictions.csv', pred);
