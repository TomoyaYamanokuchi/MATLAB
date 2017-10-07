function [optimparam, fronorm]  = uni_direct_GP(x_origin, const)



n = size(x_origin, 2);

x_train = [1:9 0];


function [f, g] = gp_regression(x, t, param)
    f = 0;
    g = 0;
    for i = n 
        gp = GP_class_image(x', t(:,i));
        [f_temp, g_temp] = gp.GaussianProcess(param);
        f = f + f_temp;
        g = g + g_temp;
    end
end


% optimization ============================================================
% theta0 = std(x_train);
% theta1 = (max(x_train) - min(x_train)) * 0.5;
% beta = 1 / (theta0 * 0.5)^2;
% param_init = [theta0, theta1, beta];
param_init = ones(1, 3) .* const;

f = @(x)gp_regression(x_train, x_origin, x); 

options = optimoptions('fminunc','Algorithm','trust-region', ...
                       'SpecifyObjectiveGradient',true);
options.Display = 'iter';
options.FiniteDifferenceType = 'central';
options.CheckGradients = true;
optimparam = fminunc(f, param_init, options);


%     options_adam = fmin_adam('defaults');
% %     options_adam.Display = 'iter';
%     options_adam.MaxFunEvals = 1e6; % colon isn't necessary for this option value
%     options_adam.TolFun = 1e-8;     % colon isn't necessary for this option value
% 
%     [optimparam] = fmin_adam(f, param_init, [],[],[],[],[],options_adam);



    


% GPregression with optimization hyper-parameters =========================
x_test = x_train;
gpr = zeros(size(x_origin));
gp = cell(1, n);
x_size = length(x_test);

for i=1:n
    gp{i} = GP_class_image(x_train', x_origin(:,i));
    gp{i}.set(optimparam);
end


tic; 
for j=1:x_size
    for i=1:n
        gpr(j, i) =  gp{i}.predict(x_test(j));
    end
end
toc;


% squared error --------------------------------------------------
fronorm =  norm(x_origin - gpr, 'fro');


% show figure =============================================================

a1 = 1;
a2 = 3;
b = 4;

figure;
os_size = sqrt(n);
width = 4;

for i=1:length(x_test)
if i<6 
    subplot(5, width, a1 + b*(i-1));  
    image(reshape(x_origin(i,:), os_size, os_size)*255);
    title('Original');
    
    subplot(5,width, a1 + b*(i-1)+1);  
    image(reshape(gpr(i,:), os_size, os_size)*255);
    title('GPR');
else
    j = i-5;
    subplot(5,width, a2 + b*(j-1));  
    image(reshape(x_origin(i,:), os_size, os_size)*255);
    title('Original');
    
    subplot(5,width, a2 + b*(j-1)+1);  
    image(reshape(gpr(i,:), os_size, os_size)*255);
    title('GPR');
end
end



end
















