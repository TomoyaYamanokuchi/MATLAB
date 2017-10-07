classdef GPLVM_Regression_x < handle
    properties
       kernel;
       beta;
       x, t;
       cov, precision;
       pre_dot_t;
    end
    methods
        function self = GPLVM_Regression_x(kernel, beta)
           self.kernel = kernel;
           self.beta = beta;
        end
        
        % calculate covariance function
        function fit(self, x, t)
           self.x = x;  
           self.t = t;  
           N = size(x, 1);
           
           % generate identity matrix that is same size of gram matrix
           I = eye(size(self.x, 1));
           
           % covariance
           self.cov = zeros(N, N);
           for n=1:N
               for m =1:N
                   xn = x(n,:);
                   xm = x(m,:);
                   self.cov(n,m) = self.kernel.calculate(xn, xm);
               end
           end
           self.cov = self.cov + (I/self.beta);
           self.precision = inv(self.cov); 
           
           % vlaue of prediction
           self.pre_dot_t = self.precision*self.t;
        end
        
        
        % log-likelihoot function of parameters ƒÆ
        function log_like = log_likelihoot(self, t)
            N = size(t, 1);
            D = size(t, 2);
            log_like = -0.5*(D*N*log(2*pi) + D*log(det(self.cov)) + trace(self.precision*t*t'));
        end
        
        
        % differentiate log likelihoo
        function grad = gradient_log_likelihood(self, x, t)
            % regression with current kernel parameters
            self.fit(x, t);
            
            % gradient dL/dC ------------------
            D = size(t, 2);
            N = size(x, 1);
            gradLC = self.precision*t*t'*self.precision - D*self.precision;
            
            
            % gradient dC/dx_nj ---------------

            Dx = size(x, 2); % dimentional of x
            DxN = Dx * N;   % total number of X
            gradCx = zeros(N, N, DxN);
            s = 1;
            for i=1:N 
                for j=1:Dx
                    for n=1:N
                        for m=1:N
                        xn = x(n,:);
                        xm = x(m,:);
                        gradCx(n,m,s) = self.kernel.grad_x(xn', xm', n, m, Dx, i, j);
                        end
                    end
                    s = s + 1;
                end
            end

            
            % gradient dL/dx ----------------
            gradLx = zeros(N, Dx);
            s = 1;
            for n=1:N
               for j=1:Dx
                   gradLx(n,j) = trace(gradLC * gradCx(:,:,s));
                   s = s + 1;
               end
            end
            
            grad{4} = reshape(gradLx, 1, []);
            
        end
        
         % predict conditional distribution to new data -> p(tn+1|t)
        function mean = predict_dist(self, x)
            k = zeros(length(self.t), 1);
            for n=1:length(self.t)
                xn = self.x(n,:);
                k(n) = self.kernel.calculate(xn, x);  % gram matrix
            end
            
            c = self.kernel.calculate(x, x) + 1/self.beta; %covariance function
            
            % predict mean and valiance
            mean = k'*self.pre_dot_t;  % (6.66) 
%             var = diag(c - k'*self.precision*k) ;  % (6.67)
        end
    end
end