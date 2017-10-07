classdef GPLVM_Regression_ver2 < handle
    properties
       kernel;
       beta;
       x, t, tt;
       cov, precision;
       precision_times_t;
    end
    methods
        function self = GPLVM_Regression(kernel, beta)
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
           self.cov = self.kernel.calculate(x);
           self.cov = self.cov + (I/self.beta);
           self.precision = self.cov\eye(size(self.cov)); % = inv(self.cov)
           self.tt = t*t';
           self.precision_times_t = self.precision*self.t;
        end
        
        
        % log-likelihoot function of parameters ƒÆ
        function log_like = log_likelihoot(self, t)
            N = size(t, 1);
            D = size(t, 2);
            log_like = -0.5*(D*N*log(2*pi) + D*logdet(self.cov) ...
                        + fast_trace_AB(self.precision, self.tt));
        end
        
        
        % differentiate log likelihoo
        function grad = gradient_log_likelihood(self, x, t)
            % regression with current kernel parameters
            self.fit(x, t);
            
            
            % gradient dL/dC ------------------
            D = size(t, 2);
            N = size(x, 1);
            gradLC = self.precision*self.tt*self.precision - D*self.precision;
            
            
            % gradient dC/dx_nj ---------------
            Dx = size(x, 2); % dimentional of x
            gradCx = zeros(N, N, N, Dx);
            for j=1:Dx
               gradCx(:,:,:,j) = self.kernel.grad_Cx(x, Dx, j);
            end
            
            
            % gradient dL/dx ----------------
            gradLx = zeros(N, Dx);
            for j=1:Dx
               gradLx(:,j) = fast_trace(gradLC .* gradCx(:,:,:,j));
            end
            

            % gradient dC/dƒÆ -----------------
            [grad_alpha, grad_gamma] = self.kernel.grad_cov(x);
            grad_beta = - eye(size(x, 1)) / self.beta^2;
            grad_cov = {grad_alpha grad_gamma grad_beta};
            
            
            % differentiate log likelihoot of parameters ƒÆ 
            grad = cell(1, 4);
            for i=1:length(grad_cov)
               grad{i} = fast_trace_AB(gradLC, grad_cov{i});
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
            mean = k'*self.precision_dot_t;  % (6.66) 
            % var = diag(c - k'*self.precision*k) ;  % (6.67)
        end
        
    end
end


function y = fast_trace_AB(A, B)
   y = sum(sum(A'.*B)); 
end

function y = fast_trace(S)
    y = sum(sum(S));
end

function v = logdet(A)
    [U, p] = chol(A);
%     if p > 0    
%         self.cov(1:size(self.cov,1)+1:end) = 1e-13;
%         cholcov = chol(self.cov);
%     end
    if p == 0
        v = 2*sum(log(diag(U)));
    else
        [~, U, P] = lu(A);
        diagU = diag(U);
        c = det(P) * prod(sign(diagU));
        v = log(c) + sum(log(abs(diagU)));
    end
end


