function data = AE_lean_and_show(x_train, l)

% training Auto-Encoder ---------------------------------------------------

[net, param] = TwoLayerNet(x_train, l); % GP

f= @(x)autofunction(x, x_train, l);

options_adam = fmin_adam('defaults');
options_adam.Display = 'iter';
options_adam.MaxFunEvals = 1e6; % colon isn't necessary for this option value
options_adam.TolFun = 1e-8;     % colon isn't necessary for this option value


x_optim = fmin_adam(f, param, [],[],[],[],[], options_adam);
x_optim = x_optim';


% setting of Auto-Encoder for Gaussian Process ----------------------------
% get optimization weights and biases 
[n, l] = size(net.W1);
[l, m] = size(net.W2);

W1 = x_optim(1:n*l);
W1 = reshape(W1, n, l);
W1_index = n*l;

B1 = x_optim(W1_index+1:(W1_index)+l);
B1_index = (W1_index)+l;

W2 = x_optim(B1_index+1:(B1_index)+l*m);
W2 = reshape(W2, l, m);
W2_index = (B1_index)+l*m;

B2 = x_optim(W2_index+1:(W2_index)+m);


% set weights and biases
net.set_param(W1, 'W1');
net.set_param(B1, 'B1');
net.set_param(W2, 'W2');
net.set_param(B2, 'B2');


% get low-dimentional representation
x_train_size = size(x_train, 1);
low = net.get_low_dim(x_train); 


% get restored image
x_rest = zeros(x_train_size, m);
for i=1:x_train_size
    x_rest(i,:) = net.predict(x_train(i,:));
end

data = {x_optim x_train x_rest low};


% show graph
a1 = 1;
a2 = 4;
b = 6;

os_resize = sqrt(l); % One Side of a resize image
os_size = sqrt(size(x_train, 2));

figure(1);
length = 5;
width = 6;
for i=1:10 %------------------
if i<6 
    subplot(length, width, a1 + b*(i-1));  
    image(reshape(x_train(i,:), os_size, os_size)*255);
    title('Original');
    
    subplot(length,width, a1 + b*(i-1)+1);  
    image(24, 24, reshape(x_rest(i,:), os_size, os_size)*255);
    title('Restored');
    
    subplot(length,width, a1 + b*(i-1)+2);  
    image(reshape(low(i,:), os_resize, os_resize)*255);
    title('Low-Dim');
else
    j = i-5;
    subplot(length,width, a2 + b*(j-1));  
    image(reshape(x_train(i,:), os_size, os_size)*255);
    title('Original');
    
    subplot(length,width, a2 + b*(j-1)+1);  
    image(24, 24, reshape(x_rest(i,:), os_size, os_size)*255);
    title('Restored');
    
    subplot(length,width, a2 + b*(j-1)+2);  
    image(reshape(low(i,:), os_resize, os_resize)*255);
    title('Low-Dim');
end
end 




end