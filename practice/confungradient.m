function [c, ceq, gc, gceq] = confungradient(x)
%CONFUNGRAD Nonlinear inequality constraints and their gradients.
% Documentation example.
c(1) = 1.5 + x(1) * x(2) - x(1) - x(2); % Inequality constraints
c(2) = -x(1) * x(2)-10; 
%   Copyright 1990-2008 The MathWorks, Inc.
ceq = [];
%c = [1.5 + x(1)*x(2) - x(1) - x(2); 
 %    -x(1)*x(2) - 10];
% Gradient (partial derivatives) of nonlinear inequality constraints:
if nargout > 2
    gc = [x(2)-1, -x(2); 
         x(1)-1, -x(1)];
% no nonlinear equality constraints (and gradients)
    gceq = [];
end 