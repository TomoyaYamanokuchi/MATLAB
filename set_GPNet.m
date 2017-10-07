function network = set_GPNet(x, X, m, gamma)

n = size(X, 2);
l = n;

% wb_size = n*m*2 + m + l;
% x = opt(1:wb_size);
%    
import GPNet
network = GPNet(X, m, gamma);

% reshape param -----------------------
W1 = x(1:n*m);
W1 = reshape(W1, n, m);
W1_index = n*m;

B1 = x(W1_index+1:(W1_index)+m);
B1 = reshape(B1, 1, []);

B1_index = (W1_index)+m;

W2 = x(B1_index+1:(B1_index)+m*l);
W2 = reshape(W2, m, l);
W2_index = (B1_index)+m*l;

B2 = x(W2_index+1:(W2_index)+l);
B2 = reshape(B2, 1, []);

% set param ---------------------------
network.set_param(W1, 'W1');
network.set_param(B1, 'B1');
network.set_param(W2, 'W2');
network.set_param(B2, 'B2');

end