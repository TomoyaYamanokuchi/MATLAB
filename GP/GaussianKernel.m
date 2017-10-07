classdef GaussianKernel < handle
   properties
       theta
   end
   methods
       function self = GaussianKernel(theta)
           assert(length(theta) == 2);
           self.theta = theta;
       end
       
       function theta = get_theta(self)
          theta = self.theta; 
       end
       
       % calculate value of kernel
       function k = calculate(self, xn, xm)
           k = self.theta(1) * exp(-0.5*self.theta(2)*(xn-xm).^2); % (6.63)
       end
       
       % differentiate CN with respect to ƒÆi 
       function grad = grad_cov(self, xn, xm, beta)
          grad1 = exp(-0.5 * self.theta(2) * (xn - xm).^2);     % İCN/İƒÆ0
          grad2 = (-0.5*self.theta(1)*(xn - xm).^2) .* grad1;   % İCN/İƒÆ1
          grad3 = - eye(length(xn)) / beta^2;
          grad = {grad1 grad2 grad3};
       end
   end
end