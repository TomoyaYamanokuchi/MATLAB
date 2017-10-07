function [optimparam, fronorm] = GP_learn_and_show_param_separate(optimized, x_origin, fmin)

wb_optim = optimized{1};
m = optimized{3};
gamma = optimized{4};

% get low-dim using optimization parameter W and B ======================== 
net = set_GPNet(wb_optim, x_origin, m, gamma);
[~, low] = net.predict(x_origin);


% Gaussian Process ========================================================
x_train_gp = [1 2 3 4 5 6 7 8 9 0];
[x_train_size, ~] = size(low);


function [f, g] = sum_loglike_param_separate(x, low, param)
    param = reshape(param, 3, m)';
    f = 0;
    g = zeros(size(param));
    for i = 1:m
        gp = GP_class_image(x' ,low(:,i));
        [f_temp, g(i,:)] = gp.GaussianProcess(param(i,:));
        f = f + f_temp;
    end
    g = reshape(g', 1, []);
end


% optimization ============================================================
% theta0 = std(x_train_gp);
% theta1 = (max(x_train_gp) - min(x_train_gp)) * 0.5;
% beta = 1 / (theta0 * 0.5)^2;
% param_init = [theta0, theta1, beta];
param_init = ones(1, 3) .* 100;
param_init = repmat(param_init, 1, m);
f = @(x)sum_loglike_param_separate(x_train_gp, low, x); 

if strcmp(fmin, 'fminunc') == 1
    options = optimoptions('fminunc','Algorithm','trust-region', ...
                           'SpecifyObjectiveGradient',true);
    options.Display = 'iter';
    options.FiniteDifferenceType = 'central';
    options.CheckGradients = true;
    optimparam = fminunc(f, param_init, options);
elseif strcmp(fmin, 'fmin_adam') == 1
    options_adam = fmin_adam('defaults');
    options_adam.Display = 'iter';
    options_adam.MaxFunEvals = 1e6;
    options_adam.TolFun = 1e-8;    
    optimparam = fmin_adam(f, param_init, [],[],[],[],[], options_adam);
else
   msg = 'fmin type is not correct!';
   error(msg);
end


% GPregression with optimization hyper-parameters =========================
x_test_gp = x_train_gp;
gpr = zeros(x_train_size, m);
optimparam = reshape(optimparam, 3, m)';

for j=1:x_train_size
    for i=1:m
        gp = GP_class_image(x_train_gp', low(:,i));
        gpr(j, i) =  gp.predict(x_test_gp(j), optimparam(i,:));
    end
end

gpr = gpr

% GPR Decode ==============================================================
gpr_decode = net.decode(gpr);


% squared error --------------------------------------------------
fronorm =  norm(x_origin - gpr_decode, 'fro');


% show figure =============================================================

a1 = 1;
a2 = 5;
b = 8;

figure;
os_resize = sqrt(m); % One Side of a resize image
os_size = sqrt(size(x_origin, 2));


for i=1:x_train_size
if i<6 
    subplot(5,8, a1 + b*(i-1));  
    image(reshape(x_origin(i,:), os_size, os_size)*255);
    title('Original');
    
    subplot(5,8, a1 + b*(i-1)+1);  
    image(reshape(gpr_decode(i,:), os_size, os_size)*255);
    title('GPR Decode');
    
    subplot(5,8, a1 + b*(i-1)+2);  
    image(reshape(low(i, :), os_resize, os_resize)*255);
    title('Low-Dim');
    
    subplot(5,8, a1 + b*(i-1)+3);  
    image(reshape(gpr(i,:), os_resize, os_resize)*255);
    title('GPR');
else
    j = i-5;
    subplot(5,8, a2 + b*(j-1));  
    image(reshape(x_origin(i,:), os_size, os_size)*255);
    title('Original');
    
    subplot(5,8, a2 + b*(j-1)+1);  
    image(reshape(gpr_decode(i,:), os_size, os_size)*255);
    title('GPR Decode');
    
    subplot(5,8, a2 + b*(j-1)+2);  
    image(reshape(low(i,:), os_resize, os_resize)*255);
    title('Low-Dim');
    
    subplot(5,8, a2 + b*(j-1)+3);  
    image(reshape(gpr(i,:), os_resize, os_resize)*255);
    title('GPR');
end



end

end

