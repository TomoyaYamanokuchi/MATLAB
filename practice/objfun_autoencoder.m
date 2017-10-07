function L = objfun_autoencoder(network, n, l, m, X) 

%--- acquire network information ------
W1 = network(1:n*l);
W1 = reshape(W1, n, l);
W1_index = n*l;

B1 = network(W1_index+1:(W1_index)+l);
B1_index = (W1_index)+l;

W2 = network(B1_index+1:(B1_index)+l*m);
W2 = reshape(W2, l, m);
W2_index = (B1_index)+l*m;

B2 = network(W2_index+1:(W2_index)+m);


%--- forward propagation ---------------

% A1 = X*W1 + B1;
% Z1 = relu(A1);
% A2 = Z1*W2 + B2;
% Y = relu(A2);

%Y = relu(relu(X*W1 + B1)*W2 + B2);
Y = 1 ./ (1 + exp(-((1 ./ (1 + exp(-(X*W1 + B1))))*W2 + B2)));
%size(Y);


%--- square sum error ------------------ 

%L = mean_squared_error(Y, X);
%L = (1/2)*(1/N)*sum(sum(((Y-X).^2)'));

L = norm(Y-X, 'fro'); % + norm(W1, 'fro') + norm(B1, 'fro') + norm(W2, 'fro') + norm(B2, 'fro');



end