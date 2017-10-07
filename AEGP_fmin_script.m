function [optimparam, fval, exitflag,output] = AEGP_fmin_script(X, m, gamma)

[two, wb_init] = GPNet(X, m, gamma);

% x_train_gp = [1:9 0];
% theta0 = std(x_train_gp);
% theta1 = (max(x_train_gp) - min(x_train_gp)) * 0.5;
% beta = 1 / (theta0 * 0.5)^2;
% gp_param_init = [theta0, theta1, beta];
gp_param_init = ones(1, 3).*200;

opt = [wb_init gp_param_init];
wb_size = length(wb_init);

optimparam = cell(1, 4); 
optimparam(1,3:4) = {m gamma};



    f= @(x)autofunctionAEGP(x, X, m, gamma);

%     options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true);
%     options.Display = 'iter';
%     options.CheckGradients = true;
%     options.FiniteDifferenceType = 'central';
%     
%     [temp, fval,exitflag,output] = fminunc(f, opt, options);
%     optimparam(1,:) = {temp(1:wb_size) temp(wb_size+1:end)};
    
    options_adam = fmin_adam('defaults');
%     options_adam.Display = 'iter';
    options_adam.MaxFunEvals = 1e6; % colon isn't necessary for this option value
    options_adam.TolFun = 1e-8;     % colon isn't necessary for this option value

    [temp, fval,exitflag,output] = fmin_adam(f, opt, [],[],[],[],[],options_adam);
    optimparam(1,1:2) = {temp(1:wb_size)' temp(wb_size+1:end)'};
    
    

end



 


