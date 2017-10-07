function f = objfun(x)
%u = 4*x(1)^2 + 2*x(2)^2 + 4*x(1)*x(2) + 2*x(2) + 1;
x1 = x(1:6)
x2 = x(7:12)
x1 = reshape(x1, 3, 2)
x2 = reshape(x2, 2, 3)
%f = exp(x(1))*(4*x(1)^2 + 2*x(2)^2 + 4*x(1)*x(2) + 2*x(2) + 1);
%f = exp(x1)*(4*x1.^2 + 2*x2.^2 + 4*x1*x2 + 2*x2 + 1);
f = det(x1*x2)
%w()

%y = 
%f = (1/2)*sum((y-t)^2)
end