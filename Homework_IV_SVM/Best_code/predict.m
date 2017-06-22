function label = predict(X, model)
m = size(X, 1);     %�������ݵĸ���
label = zeros(m,1);
for i = 1:m
    x = X(i,:);
    label(i,1)=decision(x, model);
end

function ylabel = decision(x, model) 
sv = model.SVs;      %֧������
b = -model.rho;      %������
w = model.sv_coef;   %֧������ϵ��
L = model.Label; %��ǩ
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

