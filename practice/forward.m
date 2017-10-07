function y = forward(network, x)
W1 = network{1};
b1 = network{2};
W2 = network{3};
b2 = network{4};

%r = b1
%t = W1
a1 = x*W1 + b1;
z1 = sigmoid(a1);
a2 = z1*W2 + b2;
y = identity(a2);

histogram(z1, 50);
    
end