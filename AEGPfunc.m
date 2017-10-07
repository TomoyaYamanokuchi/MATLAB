function [f, g] = AEGPfunc(opt, X, m, i)
% AE
d = size(X, 2);
wb_size = (d*m)*2 + m + d;
wb = opt(1:wb_size);
% wb = wb_init;

[fAE, gAE, low_dim] = autofunctionAE(wb, X, m);


% GP
param = opt(wb_size+1:end);
% param = opt;
% param = [ 3.0277    4.5000    0.4364 ];

x = [1:9 0];

[fGP, gGP] = autofunctionGP(x', low_dim(:,i), param);

% 

    f = fAE + fGP;
    g = [gAE gGP];
 
%     
%     f = fGP ;
%     g = gGP ;
%     
end