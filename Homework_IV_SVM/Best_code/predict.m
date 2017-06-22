function label = predict(X, model)
m = size(X, 1);     %测试数据的个数
label = zeros(m,1);
for i = 1:m
    x = X(i,:);
    label(i,1)=decision(x, model);
end

function ylabel = decision(x, model) 
sv = model.SVs;      %支持向量
b = -model.rho;      %常数项
w = model.sv_coef;   %支持向量系数
L = model.Label; %标签
len = length(model.sv_coef); 
gamma = model.Parameters(4); 
y = 0;
for i = 1: len 
    xi = sv(i,:);
    y = y + w(i) * (exp(-gamma.* sum((x-xi).^2)));
end
y = y + b;
if y >= 0 
    ylabel = L(1);
else ylabel = L(2);
end

