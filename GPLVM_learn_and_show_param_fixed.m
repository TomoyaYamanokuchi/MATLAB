function opt = GPLVM_learn_and_show_param_fixed(optimized, x_origin, fmin, const, x_init)

wb_optim = optimized{1};
m = optimized{3};
gamma = optimized{4};


% get low-dim using optimization parameter W and B ======================== 
net = set_GPNet(wb_optim, x_origin, m, gamma);
[~, low] = net.predict(x_origin);


% Gaussian Process ========================================================
% x_train_gp = [1 2 3 4 5 6 7 8 9 0];
% x_train_gp = [2 3 4 5 6 7 8 9 0];

[x_train_size, ~] = size(low);


% total log-likelihood parameteres are fixed --------------
gplvm = cell(1, m);
function [f, g] = sum_loglike_param_fixed(x, low)
    param = x(1:length(param_init));
    x = reshape(x(length(param_init)+1:end), [], dim);
    f = 0;
    g = zeros(1, length(param)+(size(x, 2)*size(x,1)));
%     g = zeros(1, length(param)+length(x))
    for i = 1:m
        gplvm{i} = GPLVM_Main(x, low(:,i));
        [f_temp, g_temp] = gplvm{i}.GaussianProcess(param);
        g = g + g_temp;
        f = f + f_temp;
    end
end


% optimization ============================================================
% theta0 = std(x_train_gp);
% theta1 = (max(x_train_gp) - min(x_train_gp)) * 0.5;
% beta = 1 / (theta0 * 0.5)^2;
% param_init = [theta0, theta1, beta];
param_init = ones(1, 3) .* const;

dim = 2;
x_init = reshape(randn(size(x_origin, 1), dim), 1, []);
% x_init = reshape(x_init, 1, []);


f = @(x)sum_loglike_param_fixed(x, low); 

if strcmp(fmin, 'fminunc') == 1
    options = optimoptions('fminunc','Algorithm','trust-region', ...
                           'SpecifyObjectiveGradient',true);
    options.Display = 'iter';
%     options.FiniteDifferenceType = 'central';
%     options.CheckGradients = true;
    optimparam = fminunc(f, [param_init x_init], options);
elseif strcmp(fmin, 'fmin_adam') == 1
    options_adam = fmin_adam('defaults');
%     options_adam.Display = 'iter';
    options_adam.MaxFunEvals = 1e6;
    options_adam.TolFun = 1e-8;    
    optimparam = fmin_adam(f, [param_init x_init], [],[],[],[],[], options_adam);
else
   msg = 'fmin type is not correct!';
   error(msg);
end

optimparam = {optimparam(1:length(param_init)) reshape(optimparam(length(param_init)+1:end), [], dim)};

% GPregression with optimization hyper-parameters =========================
x_test_gp = optimparam{2};

gpr = zeros(x_train_size, m);


% set class
gplvm = cell(1, m);
for i=1:m
    gplvm{i} = GPLVM_Main(x_test_gp, low(:,i));
    gplvm{i}.set(optimparam{1});
end

tic; 
% predict
for j=1:x_train_size
    for i=1:m
        gpr(j, i) =  gplvm{i}.predict(x_test_gp(j,:));
    end
end

% gpr

% GPR Decode ==============================================================
gpr_decode = net.decode(gpr);

toc;


% squared error --------------------------------------------------
fronorm =  norm(x_origin - gpr_decode, 'fro');

opt = [optimparam {fronorm}];



%show figure =============================================================

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

