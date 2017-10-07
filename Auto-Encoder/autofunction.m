function [f, g] = autofunction(x, data, l)


n = size(data, 2);
m = n;

import TwoLayerNet;
network = TwoLayerNet(data, l);

% reshape param -----------------------
W1 = x(1:n*l);
W1 = reshape(W1, n, l);
W1_index = n*l;

B1 = x(W1_index+1:(W1_index)+l);
B1 = reshape(B1, 1, []);
% size(B1)
B1_index = (W1_index)+l;

W2 = x(B1_index+1:(B1_index)+l*m);
W2 = reshape(W2, l, m);
W2_index = (B1_index)+l*m;

B2 = x(W2_index+1:(W2_index)+m);
B2 = reshape(B2, 1, []);
% size(B2)

% set param ---------------------------
network.set_param(W1, 'W1');
network.set_param(B1, 'B1');
network.set_param(W2, 'W2');
network.set_param(B2, 'B2');


% calculate loss ----------------------- 
f = network.loss(data, data);

% get gradient -------------------------
%g  = network.numerical_gradient(data, data, network);
g = network.analytical_gradient(data, data);

gW1 = reshape(g{1}, 1, n*l);
gB1 = g{2};
gW2 = reshape(g{3}, 1, l*m);
gB2 = g{4};

g = horzcat(gW1, gB1, gW2, gB2);

end
 
