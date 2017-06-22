X=csvread('train_data.csv');
y=csvread('train_targets.csv');
Xt=csvread('test_data.csv');
yt=csvread('test_targets.csv');

gamma_list=[0.1,1,10];
passed=0;
for i=1:length(gamma_list)
    gamma=gamma_list(i);
    model=svmtrain(y,X,['-s 0 -t 2 -c 1 -q -g ',num2str(gamma)]);

    y_pred_true=svmpredict(yt,Xt,model,'-q');

    %=========================================================
    % You should implement the "predict" function in predict.m
    % We will call it in the same path 
    y_pred=predict(Xt,model);
    %=========================================================

    passed=passed+sum(y_pred==y_pred_true)/size(y_pred_true,1);
end
if passed==length(gamma_list)
    fprintf('passed\n');
else
    fprintf('failed\n');
end
