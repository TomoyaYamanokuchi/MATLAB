function opt = GPLVM_script(x_train, t_train, const)

    function [f, g] = gplvmfunc(x, t)
        param = x(1:3);
        x = x(4:end);
        x = reshape(x, size(t,1), []);
        [f, g] = GPLVM(x, t, param);
    end

    theta = ones(1, 3)*const;
%     theta = [0.225 7.74 128];
    
    param_init = [theta reshape(x_train, 1, [])];
%     param_init = theta(1);
%     param_init = theta;
%     param_init = [reshape(x_train, 1, [])];
    
%     f = @(a)gplvmfunc([theta a] , t_train);
%     f = @(a)gplvmfunc([ a reshape(x_train, 1, [])] , t_train);
    f = @(a)gplvmfunc(a , t_train); 
    % optimization setting ----------------------------------------------                  
%     options = optimoptions('fminunc', 'Algorithm','trust-region', 'SpecifyObjectiveGradient',true);
    options = optimoptions('fminunc','SpecifyObjectiveGradient',true);
    options.Display = 'iter';
    options.MaxIterations = 1e6;
%     options.FiniteDifferenceType = 'central';
%     options.CheckGradients = true;
    optimparam = fminunc(f, param_init, options);
    
    opt1 = optimparam(1:3);
    opt2 = reshape(optimparam(4:end), size(x_train, 1), []);
    opt = {opt1 opt2};


end