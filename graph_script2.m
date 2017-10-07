function graph_script2(param, n, l, m, X) 

%--- acquire param information --------
W1 = param(1:n*l);
W1 = reshape(W1, n, l);
W1_index = n*l;

% figure(2);
% image(reshape(W1(:, 1), 28, 28)*255);

B1 = param(W1_index+1:(W1_index)+l);
B1_index = (W1_index)+l;

W2 = param(B1_index+1:(B1_index)+l*m);
W2 = reshape(W2, l, m);
W2_index = (B1_index)+l*m;

B2 = param(W2_index+1:(W2_index)+m);
%------------------------------------------

a1 = 1;
a2 = 3;
b = 4;

net = TwoLayerNet(X(1, :), l); % for initialization of wights and biases
net.set_param(W1, 'W1');
net.set_param(B1, 'B1');
net.set_param(W2, 'W2');
net.set_param(B2, 'B2');

figure(1);
for i=1:10 %------------------
    
if i<6 
    Y = net.predict(X(i,:));
    subplot(5,4, a1 + b*(i-1));  
    image(reshape(X(i,:), 28, 28)*255);
    title('Original Image');
    subplot(5,4, a1 + b*(i-1)+1);  
    image(24, 24, reshape(Y, 28, 28)*255);
    title('Restored Image');
else
    Y = net.predict(X(i,:));
    j = i-5;
    subplot(5,4, a2 + b*(j-1));  
    image(reshape(X(i,:), 28, 28)*255);
    title('Original Image');
    subplot(5,4, a2 + b*(j-1)+1);  
    image(24, 24, reshape(Y, 28, 28)*255);
    title('Restored Image');
end
end    %------------------
end
