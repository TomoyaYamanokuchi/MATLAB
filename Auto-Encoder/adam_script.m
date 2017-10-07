function [x, fval, exitflag,output] = adam_script(X, l)

[two, param] = TwoLayerNet(X, l);

f= @(x)autofunction(x, X, l);

options_adam = fmin_adam('defaults');
options_adam.Display = 'iter';
options_adam.MaxFunEvals = 1e6; % colon isn't necessary for this option value
options_adam.TolFun = 1e-8;     % colon isn't necessary for this option value
    
[x,fval,exitflag,output] = fmin_adam(f, param, [],[],[],[],[],options_adam);
x = x';
end



 


