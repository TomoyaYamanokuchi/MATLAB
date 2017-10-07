function opt = GPLVM_script_ver1(x_train, t_train, const)


    theta = ones(1, 3)*const;
    param_init = [theta reshape(x_train, 1, [])];
    
    f = @(a)gplvmfunc(a , t_train);               
    % optimization setting ----------------------------------------------                  
    options = optimoptions('fminunc', 'Algorithm','trust-region', 'SpecifyObjectiveGradient',true);
    options.Display = 'iter';
    options.FiniteDifferenceType = 'central';
    options.CheckGradients = true;
    optimparam = fminunc(f, param_init, options);
    
    
   function [f, g] = gplvmfunc(x, t)
        param = x(1:3);
        x = x(4:end);
        x = reshape(x, size(t,1), []);
        gplvm = GPLVM_Main(x, t);
        [f, g] = gplvm.GaussianProcess(param);
    end
    
    
    opt1 = optimparam(1:3);
    opt2 = reshape(optimparam(4:end), size(x_train, 1), []);
    opt = {opt1 opt2};


end