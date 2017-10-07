classdef GPLVM_Kernel_ver2 < handle
   properties
       alpha, gamma, beta
   end
   methods
       function self = GPLVM_Kernel_ver2(param)
           assert(length(param) == 2);
           self.alpha = param(1);
           self.gamma = param(2);
       end
       
       % calculate value of kernel
       function k = calculate(self, x) 
          n = size(x, 1); 
          [m1, m2] = meshgrid([1:n]', 1:n) 
          k = ones(n, n);
          adress = find(k);
          datainfo1 = reshape(m1,[],1);
          datainfo2 = reshape(m2,[],1);   
          usedata1 = x(datainfo1, :);
          usedata2 = x(datainfo2, :);
          c = self.alpha * exp(-0.5*self.gamma*norm_row(usedata1-usedata2).^2);
          k(adress) = c;
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
       
       % gradient x
       function grad = grad_Cx(self, x, Dx, j)
           
           n = size(x, 1);
           mask_unit = 1:n;
           mask = reshape(mask_unit, 1, 1, n);
           mask = repmat(mask, n, n, 1);
           [m1, m2] = meshgrid(mask_unit', mask_unit, mask_unit);
%            m1 = repmatrow(mask_unit, n);
%            m2 = repmatcolomun(mask_unit', n);
           
           
           masked1 = zeros(n, n, n);
           masked2 = zeros(n, n, n);
           
           masked1(mask==m2) = 1;
           masked2(mask==m1) = 2;
           masked1(m1==m2) = 0;
           masked2(m1==m2) = 0;
           
           
           masked1_address = find(masked1);
           
           masked1_datainfo1 = m2(masked1==1);
           masked1_datainfo2 = m1(masked1==1);           
           
           masked2_address = find(masked2);
           
           masked2_datainfo1 = m2(masked2==2);
           masked2_datainfo2 = m1(masked2==2);
           
           
           masked1_usedata1 = x(masked1_datainfo1, :);
           masked1_usedata2 = x(masked1_datainfo2, :);
           masked2_usedata1 = x(masked2_datainfo1, :);
           masked2_usedata2 = x(masked2_datainfo2, :);
           
           dxx = [zeros(1, j-1) 1 zeros(1, Dx-j)];
           k11 = -self.gamma.*(masked1_usedata1-masked1_usedata2)';
           k12 = self.alpha*(exp(-0.5*self.gamma*norm_row(masked1_usedata1-masked1_usedata2).^2))';
           k1 = dxx*(k11.*k12);
           masked1(masked1_address) = k1;
           
           k21 = self.gamma.*(masked2_usedata1-masked2_usedata2)';
           k22 = self.alpha*(exp(-0.5*self.gamma*norm_row(masked2_usedata1-masked2_usedata2).^2))';
           k2 = dxx*(k21.*k22);
           masked2(masked2_address) = k2;
           
           grad = masked1 + masked2;
           
       end
   end
end


function y = norm_row(S)
   y = sqrt(sum(S.^2, 2));
end

function m = repmatrow(v, n)
    m = repmat(v, n, 1, n);
end

function m = repmatcolomun(v, n)
    m = repmat(v, 1, n, n);
end
