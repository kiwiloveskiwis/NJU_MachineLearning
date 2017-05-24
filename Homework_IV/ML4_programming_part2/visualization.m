% created by Yuri Wu, 2017-04-23
% for SVM visualization

X=csvread('demo_data.csv');
y=csvread('demo_targets.csv');

demo_list={'-t 0 -c 1','-t 1 -d 3','-t 2 -c 1 -g 10',...
    '-t 2 -c 10 -g 10','-t 2 -c 1 -g 100','-t 2 -c 1 -g 100'};

titles = {'linear kernel',
          'polynomial (degree 3) kernel',
          'RBF kernel, gamma=10, C=1',
          'RBF kernel, gamma=10, C=10',
          'RBF kernel, gamma=100, C=1',
          'RBF kernel, gamma=100, C=10'};

x_min=0;y_min=0;x_max=1;y_max=1;
[xx,yy] = meshgrid(linspace(x_min,x_max,400)',linspace(y_min,y_max,400)');

map = [ 167, 180, 236; 229, 160, 156];
map=map/255;
colormap(map);

for i=1:length(demo_list)
    para=demo_list{i};
    model=svmtrain(y,X,['-s 0 -q ',para]);
    
    Z = zeros(size(xx(:),1),1);
    Z=svmpredict(Z,[xx(:),yy(:)],model,'-q');
    Z=reshape(Z,size(xx));
    
    fig=subplot(3,2,i);
    contourf(xx, yy, Z);
    axis([x_min x_max y_min y_max]);
    
    %contourcmap('lines');
    hold on;
    gscatter(X(:,1),X(:,2),y,'br','..',30);
    legend(fig,'off');
    title(fig,titles{i});
    
end







