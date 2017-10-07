function [optimparam] = AEGPLVM_fmin_script(X, m, gamma, x_gplvm, const)

[two, wb_init] = GPLVMNet(X, m, gamma);

% x_train_gp = [1:9 0];
% theta0 = std(x_train_gp);
% theta1 = (max(x_train_gp) - min(x_train_gp)) * 0.5;
% beta = 1 / (theta0 * 0.5)^2;
% gp_param_init = [theta0, theta1, beta];
gp_param_init = ones(1, 3).*const;


% GPLVM input data
dim = 2;
% x_gplvm = randn(size(X, 1), dim)*1e-2;


opt = [wb_init gp_param_init reshape(x_gplvm, 1, [])];
wb_size = length(wb_init);
gplvm_param_end = wb_size + length(gp_param_init);


optimparam = cell(1, 5); 
optimparam(1,4:5) = {m gamma};



    f= @(x)autofunctionAEGPLVM(x, X, m, gamma, dim);

%     options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true);
%     options.Display = 'iter';
%     options.CheckGradients = true;
%     options.FiniteDifferenceType = 'central';
%     options.MaxFunEvals = 1e6; % colon isn't necessary for this option value
%     options.TolFun = 1e-8;     % colon isn't necessary for this option value
    
%     [temp] = fminunc(f, opt, options);
%     optimparam(1,:) = {temp(1:wb_size) temp(wb_size+1:end)};
    
    options_adam = fmin_adam('defaults');
    options_adam.Display = 'iter';
    options_adam.MaxFunEvals = 1e6; % colon isn't necessary for this option value
%     options_adam.TolFun = 1e-8;     % colon isn't necessary for this option value

    [temp] = fmin_adam(f, opt, [],[],[],[],[],options_adam);
    optimparam(1,1) = {temp(1:wb_size)'} ;
    optimparam(1,2) = {temp(wb_size+1:gplvm_param_end)}; 
    optimparam(1,3) = {reshape(temp(gplvm_param_end+1:end), [], dim)};
    
    

end



 


