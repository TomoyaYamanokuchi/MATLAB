function f = objfun2(x1, x2)

u = 4*x1.^2 + 2*x2.^2 + 4*x1.*x2 + 2*x2 + 1;
f = exp(x1)*(u);

%W1 = []

%y = exp(a-C) ./ sum(exp(a-C)) 
%f = (1/2)*sum((y)^2)
end