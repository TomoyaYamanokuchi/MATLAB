function [x, fval, exitflag,output] = GP_script_mnist(X)


% set data 
data = cell(1, 10);
data{1} = X(4, :)';
% size(data{1})
data{2} = X(6, :)';
data{3} = X(8, :)';
data{4} = X(10, :)';
data{5} = X(36, :)';
data{6} = X(14, :)';
data{7} = X(16, :)';
data{8} = X(18, :)';
data{9} = X(5, :)';
data{10} = X(2, :)';
% image(reshape(data{10}, 28, 28)*255);





% t_train = [data{1} data{2} data{3}, data{4} data{5}]';
% % size(t_train)
% x_train = (1:size(t_train, 1))';
% x_test = (size(t_train,1)+1:length(data))';
% 
% data = {x_train t_train x_test};



% Gausian Prosess -----------------------------------------------------
gp = GP_class_image(data);

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