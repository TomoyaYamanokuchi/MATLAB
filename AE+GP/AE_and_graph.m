function data = AE_and_graph(X, l)

N = 10;
D = 784;
x_train = zeros(N, D);

x = [4 6 8 10 36 14 16 18 5 2];
for i=1:N
   x_train(i,:) = X(x(i),:); 
end


x = adam_script(x_train, l);

graph_script2(x, size(x_train, 2), l, size(x_train, 2), x_train);

data = {x x_train x_rest};

end