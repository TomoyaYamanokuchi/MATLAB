classdef GPRegression < handle
    properties
       kernel;
       beta;
       x, t;
       cov, precision;
    end
    methods
        function self = GPRegression(kernel, beta)
           self.kernel = kernel;
           self.beta = beta;
        end
        
        % calculate covariance function
        function fit(self, x, t)
           self.x = x;  % x = [x1 x2 ... xN]' 
           self.t = t;  % t = [t1 t2 ... tN]' 
           
           % generate identity matrix that is same size of gram matrix
           I = eye(length(self.x)); 
           [xn, xm] = meshgrid(self.x, self.x);
           self.cov = self.kernel.calculate(xn, xm) + (I/self.beta) ;  % (6.62)  
           self.precision = inv(self.cov);
        end
        
        
        % log-likelihoot function of parameters ƒÆ
        function log_like = log_likelihoot(self, t)
%             t = t'; 
            N = size(t, 1);
            log_like = -0.5 * (log(det(self.cov)) + t'*self.precision*t + N*log(2*pi)); % (6.69)
        end
        
        
        % differentiate log likelihoo
        function grad = gradient_log_likelihood(self, x, t)
            % regression with current kernel parameters
            self.fit(x, t);
            
            
            % differentiate covariance matrtix with respect to parameter ƒÆi
            [xn, xm] = meshgrid(x, x);
            grad_cov = self.kernel.grad_cov(xn, xm, self.beta);
            
            % differentiate log likelihoot of parameters ƒÆ with respect to
            % parameter ƒÆi
%             t = t';
%             size(t)
            grad = cell(1, 3);
            size(t')
            trace(self.precision*grad_cov{1})
            for i = 1:length(grad_cov)
                grad{i} = -0.5*( trace(self.precision*grad_cov{i})...
                                      - t' * self.precision * grad_cov{i} * self.precision * t ); % (6.70)
            end
        end
        
        
        % predict conditional distribution to new data -> p(tn+1|t)
        function [mean, var] = predict_dist(self, x)
            [xn, xm] = meshgrid(x, self.x);
            k = self.kernel.calculate(xn, xm);  % gram matrix
            
            % values of all c are same value  
            % so calculate a c of all c 
            c = self.kernel.calculate(x(1), x(1)) + 1/self.beta; 
                     
            mean = k'*self.precision*self.t;      % (6.66) 
            var = diag(c - k'*self.precision*k);   % (6.67)
        end
    end
end