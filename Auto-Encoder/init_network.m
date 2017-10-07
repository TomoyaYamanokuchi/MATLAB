function network = init_network(X, l)

n = size(X, 2);
m = n;

W1 = randn(n, l) ./ sqrt(n);
b1 = zeros(1, l);
W2 = randn(l, m) ./ sqrt(l);
b2 = zeros(1, m);

%--procces for hand to fminunc----
W1 = reshape(W1, 1, n*l);
W2 = reshape(W2, 1, l*m);


network = [{horzcat(W1, b1, W2, b2)} {n} {l} {m} {X}];

% if type == 0
%     network = [{horzcat(W1, b1, W2, b2)}, {n}, {l}, {m}];
% elseif type == 1
%     network = [{W1} {b1} {W2} {b2} {n} {l} {m}];
% else
%     disp('"type" number is not correct!');

end

