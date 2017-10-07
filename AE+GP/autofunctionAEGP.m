function [f, g] = autofunctionAEGP(opt, X, m, gamma)


n = size(X, 2);
l = n;

wb_size = n*m*2 + m + l;
x = opt(1:wb_size);
gp_param = opt(wb_size+1:end);


% import TwoLayerNet;             
% network = TwoLayerNet(X, m);    
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


% calculate loss -----------------------    
f = network.loss(X, X, gp_param);

% get gradient -------------------------
%g  = network.numericam_gradient(X, X, network);   
g = network.analytical_gradient(X, X, gp_param);    

gW1 = reshape(g{1}, 1, n*m);
gB1 = g{2};
gW2 = reshape(g{3}, 1, m*l);
gB2 = g{4};
gGPparam = g{5};

g = horzcat(gW1, gB1, gW2, gB2, gGPparam);


end
 