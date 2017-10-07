function opt = GPLVM_script2(x, t, const)

    function [f, g] = gplvmfunc(x, t)
        param = x(1:3);
        x = x(4:end);
        x = reshape(x, size(t,1), []);
        
        f = 0;
        g = 0;
        for i=1:size(t, 2)
            gplvm = GPLVM_Main(x, t(:,i));
            [f_temp, g_temp] = gplvm.GaussianProcess(param);
            f = f + f_temp;
            g = g + g_temp;
        end
    end



%     theta = [0.1 10 150];
      theta = ones(1, 3)*const;
%     param_init = [theta reshape(x, 1, [])];
%     ini = reshape(x, 1, []);
    ini = x;
    
    
    f = @(a)gplvmfunc(a , t); 

    options = optimoptions('fminunc','Algorithm','trust-region', ...
                           'SpecifyObjectiveGradient',true);
  
    options.FiniteDifferenceType = 'central';
    options.CheckGradients = true;
%     options = optimoptions('fminunc','Algorithm','quasi-newton');
%         options = fmin_adam('defaults');
        options.Display = 'iter';
    options.MaxFunEvals = 1e6; 
%     options.TolFun = 1e-8;  
%     options.FiniteDifferenceType = 'central';


    opt = fminunc(f, [theta reshape(ini, 1, [])], options);
%         [opt] = fmin_adam(f, [theta reshape(ini, 1, [])], [],[],[],[],[],options);
        
    opt1 = opt(1:3);
    opt2 = reshape(opt(4:end), size(x, 1), []);
    opt = {opt1 opt2};


end