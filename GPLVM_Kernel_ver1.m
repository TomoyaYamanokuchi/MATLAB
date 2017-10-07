classdef GPLVM_Kernel_ver1 < handle
   properties
       alpha, gamma, beta
   end
   methods
       function self = GPLVM_Kernel_ver1(param)
           assert(length(param) == 2);
           self.alpha = param(1);
           self.gamma = param(2);
       end
       
       % calculate value of kernel
       function k = calculate(self, xn, xm)
           k = self.alpha * exp(-0.5*self.gamma*norm(xn-xm)^2);
       end
       
       % differentiate CN with respect to ��i 
       function [grad1, grad2] = grad_cov(self, xn, xm)
          grad1 = exp(-0.5 * self.gamma * norm(xn-xm)^2);        % ��CN/�݃�
          grad2 = (- 0.5 * norm(xn-xm)^2) * self.alpha * grad1;  % ��CN/�݃�
%           grad3 = - eye(length(xn)) / beta^2   ;             % ��CN/�݃�
%           grad = {grad1 grad2 grad3};
       end
       
       % gradient x
       function grad = grad_x(self, xn, xm, n ,m, Dx, i, j)
           
           
          dxx = zeros(1, Dx);
          if n==i && m~=i
            dxx(j) = 1;
            grad = dxx*(-self.gamma*(xn-xm))*self.alpha*exp(-0.5*self.gamma*norm(xn-xm)^2);
%               grad = dxx*(-self.gamma*(xn-xm))*self.alpha*exp(-0.5*self.gamma*(xn-xm)'*(xn-xm));
          elseif n~=i && m==i
            dxx(j) = 1;
            grad = dxx*(self.gamma*(xn-xm))*self.alpha*exp(-0.5*self.gamma*norm(xn-xm)^2);
%               grad = dxx*(self.gamma*(xn-xm))*self.alpha*exp(-0.5*self.gamma*(xn-xm)'*(xn-xm));
          else
            grad = 0;
          end
       end
   end
end