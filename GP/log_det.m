function [f, g2] = log_det(param)

a = 2;
b = param;


x = [2 5 3] ;
[xn, xm] = meshgrid(x, x);

K = a*exp(-0.5 * b* (xn-xm).^2);
I = eye(length(x));
C = K + 0.01*I;


f = -log(det(C));


grad_a = exp(-0.5 * b * (xn-xm).^2);
grad_b = (-0.5 * (xn-xm).^2 * a) .* grad_a;

g1 = -trace(inv(C) * grad_a);
g2 = -trace(inv(C) * grad_b);

% g = [g1;
%      g2];


end