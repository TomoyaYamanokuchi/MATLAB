function low_graph_num(X)


figure;
scatter(X(:,1),X(:,2),17, [1 1 1], 'filled')

xt = reshape(X(:,1), 1, []);
yt = reshape(X(:,2), 1, []);
str = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'};
text(xt, yt, str, 30);


end