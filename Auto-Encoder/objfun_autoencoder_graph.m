function L = objfun_autoencoder_graph(network, n, l, m, X) 

figure(1); 
image(reshape(X, 28, 28)*255);


%--- acquire network information ------
W1 = network(1:n*l);
W1 = reshape(W1, n, l);
W1_index = n*l;

% figure(2);
% image(reshape(W1(:, 1), 28, 28)*255);

B1 = network(W1_index+1:(W1_index)+l);
B1_index = (W1_index)+l;

W2 = network(B1_index+1:(B1_index)+l*m);
W2 = reshape(W2, l, m);
W2_index = (B1_index)+l*m;

B2 = network(W2_index+1:(W2_index)+m);


%--- forward propagation ---------------

A1 = X*W1 + B1;
% figure(2);
% histogram(A1, 50);

Z1 = sigmoid_func(A1);
% figure(3);
% histogram(Z1, 50);

A2 = Z1*W2 + B2;
Y = sigmoid_func(A2);

%Y = relu(relu(X*W1 + B1)*W2 + B2);
%Y = 1 ./ (1 + exp(-(1 ./ (1 + exp(-(X*W1 + B1)))*W2 + B2)));
%size(Y);
figure(2);
imagesc(reshape(Y, 28, 28)*255);
% 
% figure(3);
% histogram(Z1, 50);

%--- square sum error ------------------ 

%L = mean_squared_error(Y, X);
N = size(Y, 1);
L = 0.5*(1/N)*sum(sum(((Y-X).^2)'));
%L = norm(Y-X, 'fro') + norm(W1, 'fro') + norm(B1, 'fro') + norm(W2, 'fro') + norm(B2, 'fro')