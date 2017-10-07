classdef GPLVM_Kernel < handle
   properties
       alpha, gamma, beta, norm_x;
   end
   methods
       function self = GPLVM_Kernel(param)
           assert(length(param) == 2);
           self.alpha = param(1);
           self.gamma = param(2);
       end
       
       % calculate value of kernel
       function k = calculate(self, x) 
           self.norm_x = norm_of_kernel(x');
           k = self.alpha * exp(-0.5*self.gamma*self.norm_x.^2);
       end
       
       % differentiate CN with respect to ƒÆi 
       function [grad_alpha, grad_gamma] = grad_cov(self, x)
           
           
          n = size(x, 1); 
          [m1, m2] = meshgrid([1:n]', 1:n); 
          grad_alpha = ones(n, n);
          grad_gamma = ones(n, n);
          adress = find(grad_alpha);
          datainfo1 = reshape(m1,[],1);
          datainfo2 = reshape(m2,[],1);   
          usedata1 = x(datainfo1, :);
          usedata2 = x(datainfo2, :);
          ga = exp(-0.5 * self.gamma * norm_row(usedata1-usedata2).^2);
          gg = (- 0.5 * norm_row(usedata1-usedata2).^2) * self.alpha .* ga;
          grad_alpha(adress) = ga;
          grad_gamma(adress) = gg;
       end
       
   end
end

function R = norm_of_kernel(S)
R1 = diag(S'*S);
R2 = R1;
R3 = -2*(S'*S);
R_2p = (R1 + R3)' + R2; 
R = sqrt(R_2p);
end


function y = norm_row(S)
   y = sqrt(sum(S.^2, 2));
end

