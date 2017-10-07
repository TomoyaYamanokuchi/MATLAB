classdef GPLVM_Regression < handle
    properties
       kernel;
       beta;
       x, t, tt;
       Kinv, K;
       precision_times_t;
    end
    methods
        function self = GPLVM_Regression(kernel, beta)
           self.kernel = kernel;
           self.beta = beta;
        end
        
        
        % calculate covariance function
        function fit(self, x, t)
           self.x = x  ;
%            f = x
           self.t = t;  
           % generate identity matrix that is same size of gram matrix
           I = eye(size(self.x, 1));
           % covariance
           self.K = self.kernel.calculate(x);
%            H = self.K 
           self.K = self.K + (I/self.beta);
%            K = self.K
           self.Kinv = self.K\eye(size(self.K)); 
%            Kinv  = self.Kinv
           
           self.tt = t*t';
%            self.precision_times_t = self.Kinv*self.t;
        end
        
        
        % log-likelihoot function of parameters ƒÆ
        function log_like = log_likelihoot(self, t)
            N = size(t, 1);
            D = size(t, 2);
            log_like = -0.5*(D*N*log(2*pi) + D*logdet(self.K) ...
                        + fast_trace_AB(self.Kinv*t, t'));
        end
        
        
        % differentiate log likelihoo
        function grad = gradient_log_likelihood(self, x, t)
            % regression with current kernel parameters
            self.fit(x, t);
            
            
            % gradient dL/dC ------------------
            [n, Dt] = size(t);
            Dx = size(x, 2);
            gradLK = self.Kinv*self.tt*self.Kinv - Dt*self.Kinv;
            
            
            % gradient dL/dx ----------------
            xdiff = permute(x, [1 3 2])-permute(x, [3 1 2]); % matrix of (xn-xm)
            gradLx = zeros(n, Dx);
            for i = 1:Dt
               Kinvt = self.Kinv*t(:,i);
               dLdK = Kinvt*Kinvt' - self.Kinv;
               for j = 1:Dx
                   gradLx(:,j) = gradLx(:,j) - self.kernel.gamma*sum(dLdK.*self.K.*xdiff(:,:,j), 2);
               end
            end
            
            
            % gradient dC/dƒÆ -----------------
            [grad_alpha, grad_gamma] = self.kernel.grad_cov(x);
            grad_beta = - eye(size(x, 1)) / self.beta^2;
            grad_cov = {grad_alpha grad_gamma grad_beta};
            
            
            % differentiate log likelihoot of parameters ƒÆ 
            grad = cell(1, 4);
            for i=1:length(grad_cov)
               grad{i} = fast_trace_AB(gradLK, grad_cov{i});
            end
            grad{4} = reshape(2*gradLx, 1, []);
            
            
        end
    end
end


function y = fast_trace_AB(A, B)
   y = sum(sum(A'.*B)); 
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

