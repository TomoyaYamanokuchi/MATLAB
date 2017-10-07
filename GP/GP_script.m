function [x, fval, exitflag,output] = GP_script()

% set data 
data = data_sin();
gp = GP_class_confirm(data);

        x_train = data{1};



% optimization 
f = @gp.GaussianProcess;

% % options = optimoptions('fminunc','Algorithm','trust-region',...
% %                       'SpecifyObjectiveGradient',true, 'CheckGradient', true);
options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true);
options.Display = 'iter';
options.FiniteDifferenceType = 'central';

% theta0 -> stddev of observation
% theta1 -> (max x - min x) / 2
% beta -> 1 /(theta0 / 2)^2

theta0 = std(x_train);
theta1 = (max(x_train) - min(x_train)) * 0.5;
beta = 1 / (theta0 * 0.5)^2;

param = [theta0, theta1, beta];
[x,fval,exitflag,output] = fminunc(f, param, options);


% GPregression with optimization hyper-parameters ----------------
gp.GaussianProcess(x);
% show predict distribution to test data
gp.show();


end