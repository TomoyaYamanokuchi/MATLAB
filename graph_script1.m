function value = graph_script1(param, n, l, m, X) 


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



% value = load('auto-encoder.mat','TrainImages168');  
% value = value.TrainImages168;

value = X;

% value1 = value(1, :);
% value6 = value(4, :);
% value8 = value(6, :);
% value1_difficult = value(9, :);

%value = {value1 value6 value8 value1_difficult};


a = 1;
b = 2;
n = size(value, 2);

net = TwoLayerNet(value(1, :), l);
net.set_param(W1, 'W1');
net.set_param(B1, 'B1');
net.set_param(W2, 'W2');
net.set_param(B2, 'B2');

% n = n;

pause('on');

figure(1);
for i=1:n

Y = net.predict(value(i, :));
subplot(1,2, 1);  

image(reshape(value(i, :), 28, 28)*255);
% axis([0 24 -0 24]);
title('Original Image')


subplot(1,2, 2);  
% figure(2);
image(24, 24, reshape(Y, 28, 28)*255);
% axis([0 24- 24 24])
title('Restored Image')
pause(1)
end



% figure
% for i=1:n
% 
% Y = net.predict(value{i});
% subplot(n,2, a + b*(i-1));  
% image(reshape(value{i}, 28, 28)*255);
% subplot(n,2, a + b*(i-1) + 1);  
% imagesc(reshape(Y, 28, 28)*255);
% end


end

