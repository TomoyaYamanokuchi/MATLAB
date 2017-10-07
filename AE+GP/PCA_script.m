function x_new = PCA_script(X, color)

x_new = X - mean(X, 2);
[U,S,V] = svd(x_new,'econ');

size(x_new)
size(V)


C = x_new*V;
size(C)

x_new = [C(:,1) C(:,2)];

figure;
scatter(x_new(:,1),x_new(:,2),17, color, 'filled')


end