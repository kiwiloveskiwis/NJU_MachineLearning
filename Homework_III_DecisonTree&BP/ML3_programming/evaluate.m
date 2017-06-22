targets=csvread('test_targets.csv');
predictions=csvread('test_predictions.csv');
acc=sum(targets==predictions)/size(targets,1);
fprintf('%f\n',acc);