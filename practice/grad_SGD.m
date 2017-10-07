function f = grad_SGD(x0, y0, lr)

alpha = 0.9;
step = 50;
x = zeros(1, step);
y = zeros(1, step);
f = zeros(1, step);
vx = zeros(1, step);
vy = zeros(1, step);

x(1) = x0;
y(1) = y0;
vx0 = 0;
vy0 = 0; 
f(1) = (1/20)*x(1)^2 + y(1)^2;

% -- SGD -------------------------------
% for i = 1:step
%     grad_x = (1/10)*x(i);
%     grad_y = 2*y(i);
%     
%     x(i+1) = x(i) - (lr*grad_x);
%     y(i+1) = y(i) - (lr*grad_y);
% %     f(i+1) = (1/20)*x(i+1)^2 + y(i)^2; 
% end

% -- Momentum -------------------------------
for i = 1:step
    grad_x = (1/10)*x(i);
    grad_y = 2*y(i);
    
    if i == 1
        vx(i) = alpha*vx0 - lr*grad_x;
        vy(i) = alpha*vy0 - lr*grad_y;
    else 
        vx(i) = alpha*vx(i-1) - lr*grad_x;
        vy(i) = alpha*vy(i-1) - lr*grad_y;
    end
    
    x(i+1) = x(i) + vx(i);
    y(i+1) = y(i) + vy(i);
end


% -- Graph -----------------------------
Xtemp = -10:.2:10;
Ytemp = -10:.2:10;
[X, Y] = meshgrid(Xtemp, Ytemp);
F = (1/20)*X.^2 + Y.^2;
mesh(X, Y, F);
contour3(X,Y,F,105)
xlabel('x');
ylabel('y');
%title('Momentum(alpha=0.9, lr=0.01, step=50)');
title('SGD(lr=0.01, step=50)');
view(2);
hold on;
%plot3(x, y, f)
plot(x, y, '-ob', 'MarkerFaceColor', 'k', 'MarkerEdgeColor','k', 'MarkerSize', 3);

end