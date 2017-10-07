function grad = numerical_gradient(s)

f = s(1).f;
x = s(2).f;
pname = s(3).f;
net = s(4).f;

h = 1e-4;    % 0.0001
grad = zeros(size(x));


for i = 1:numel(x)
    temp = x(i);
    % calculate f(x+h)
    x(i) = temp + h;
    net.set_param(x, pname);
    fxh1 = f(x);
    
    % calculate f(x-h)
    x(i) = temp - h;
    net.set_param(x, pname);
    fxh2 = f(x);
    
    grad(i) = (fxh1 - fxh2) / (2*h);
    
    % reposit value x(i)
    x(i) = temp;
    net.set_param(x, pname);
end

%grad;



end
