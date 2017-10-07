function [f, g] = GPLVM(x, t, param) 
[n, Dt] = size(t);
Dx = size(x, 2);

% set hyper parameters
alpha = param(1);
gamma = param(2);
beta = param(3);

% calculate kernel function
norm_x = norm_of_kernel(x');
K = alpha * exp(-0.5*gamma*norm_x.^2);
KK = K;
K = K + diag(repmat(1/beta, n, 1));
[v, U] = logdet(K);
% Kinv = K\eye(size(K));
Uinv = U\eye(size(U));
Kinv = Uinv*Uinv';

% gradient 
xdiff = permute(x, [1 3 2])-permute(x, [3 1 2]); % matrix of each (xn-xm)
gradLx = zeros(n, Dx);
grad_alpha = 0;
grad_gamma = 0;
grad_beta = 0;
for i = 1:Dt
   Kinvt = Kinv*t(:,i);
   gradLK = Kinvt*Kinvt' - Kinv;
   for j = 1:Dx
       gradLx(:,j) = gradLx(:,j) - gamma*sum(gradLK.*xdiff(:,:,j).*KK, 2);   
   end
end
KinvT = Kinv*t;
f = - 0.5*(Dt*n*log(2*pi) + Dt*logdet(K) + fast_trace_AB(KinvT, t'));
f = -f;
gradLK_all = KinvT*KinvT'- Dt*Kinv;
grad_alpha = grad_alpha + (1/alpha)*fast_trace_AB(gradLK_all, KK);
grad_gamma = grad_gamma + (-0.5)*fast_trace_AB(gradLK_all, norm_x.^2.*KK);
grad_beta = grad_beta + (-(1/beta^2)*trace(gradLK_all));
g =  -0.5*[grad_alpha grad_gamma grad_beta 2*reshape(gradLx, 1, [])];



% jnjnj = kmkkm
% 
% g =  0.5*[2*reshape(gradLx, 1, [])];
% g =  0.5*[grad_beta];
% g =  0.5*[grad_gamma];
% g =  0.5*[grad_alpha];
% g = 0.5*[grad_alpha grad_gamma grad_beta];
end



function R = norm_of_kernel(S)
R1 = diag(S'*S);
R2 = R1;
R3 = -2*(S'*S);
R_2p = (R1 + R3)' + R2; 
R = sqrt(R_2p);
end

function y = fast_trace_AB(A, B)
   y = sum(sum(A'.*B)); 
end


function [v, U]  = logdet(A)
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
