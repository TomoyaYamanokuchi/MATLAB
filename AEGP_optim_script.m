function param_optim =  AEGP_optim_script(X, m)


[net, wb_init] = TwoLayerNet(X, m); 



x_train_gp = [1:9 0];
theta0 = std(x_train_gp);
theta1 = (max(x_train_gp) - min(x_train_gp)) * 0.5;
beta = 1 / (theta0 * 0.5)^2;

gp_param_init = [theta0, theta1, beta];
% gp_param_init = repmat(gp_param_init, 1, m);


opt = horzcat(wb_init, gp_param_init);
% opt = wb_init;
% opt = gp_param_init;


% f= @(x)AEGPfunc(x, X, m);
% 
% options_adam = fmin_adam('defaults');
% options_adam.Display = 'iter';
% options_adam.MaxFunEvals = 1e6; % colon isn't necessary for this option value
% options_adam.TolFun = 1e-8;     % colon isn't necessary for this option value

% x_optim = fmin_adam(f, opt, [],[],[],[],[], options_adam);
% x_optim = x_optim';

wb_init_size = size(wb_init, 2);
param_optim = cell(m, 2);
for i=1:m

    f = @(x)AEGPfunc(x, X, m, i);
    options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true);
    options.Display = 'iter';
    options.CheckGradients = true;
    options.FiniteDifferenceType = 'central';

    temp = fminunc(f, opt, options);
    param_optim(i,:) = {temp(1:wb_init_size) temp(wb_init_size+1:end)}; 
end
    
end