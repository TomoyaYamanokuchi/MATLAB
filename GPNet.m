classdef GPNet < handle
    properties
       W1, B1, W2, B2, W3, B3, W4, B4, W5, B5, W6, B6;
       Affine1, Affine2, Affine3, Affine4, Affine5, Affine6;
       Sigmoid1, Sigmoid2, AddLayer;
       ReLu1, ReLu2, ReLu3, ReLu4,ReLu5, ReLu6,ReLu7;
       SquareError, LogLikelihood;
       SigmoidDualOutput;
       gamma, MullLayler;
    end
    methods
        function [self, param] = GPNet(X, m, gamma)
            n = size(X, 2); % dimension of input data
            l = n;          % dimension of output data
            self.gamma = gamma;
            
            % format for fmin optimization function
            % generation of Weights and Biases
            self.W1 = randn(n, m) ./ sqrt(n); 
            self.B1 = zeros(1, m);
            self.W2 = randn(m, l) ./ sqrt(m);
            self.B2 = zeros(1, l);
            
             
            W1 = reshape(self.W1, 1, n*m);
            W2 = reshape(self.W2, 1, m*l);
            
            param = horzcat(W1, self.B1, W2, self.B2);
            
            
            % generation of Layer ===================================

            self.Affine1 = Affine(self.W1, self.B1);
            self.Sigmoid1 = SigmoidDualOutput();
            self.Affine2 = Affine(self.W2, self.B2);
            self.Sigmoid2 = Sigmoid();
            self.SquareError = SquareError();
            self.LogLikelihood = LogLikelihood();
            self.MullLayler = MulLayer();
            self.AddLayer = AddLayer();
             
        end
        
        
        % forward propagation ----------------------------------------
        function [y, lowdim] = predict(self, x)
            % Sigmoid
            y = self.Affine1.forward(x, self.W1, self.B1);
            y = self.Sigmoid1.forward(y);
            lowdim = y;
            y = self.Affine2.forward(y, self.W2, self.B2);
            y = self.Sigmoid2.forward(y);
        end
        
        
        % get decode data --------------------------------------------
        function y = decode(self, x)
            % Sigmoid
            y = self.Affine2.forward(x, self.W2, self.B2);
            y = self.Sigmoid2.forward(y);
        end
        
        
        % calculate loss function -------------------------------------
        function L = loss(self, X, T, param)
            [Y, lowdim] = self.predict(X);
            L1 = self.SquareError.forward(Y, T);
            L2 = self.LogLikelihood.forward(lowdim, param);
            L2_gamma = self.MullLayler.forward(L2, self.gamma);
            L = self.AddLayer.forward(L1, L2_gamma);
        end


        % Numerilcal Gradient -----------------------------------------
        function grads = numerical_gradient(self, X, T, net)
            
            loss_W = @(W) self.loss(X, T);
            
            grad_W1 = numerical_gradient(struct('f', {loss_W, self.W1, 'W1', net}));
            grad_B1 = numerical_gradient(struct('f', {loss_W, self.B1, 'B1', net}));
            grad_W2 = numerical_gradient(struct('f', {loss_W, self.W2, 'W2', net}));
            grad_B2 = numerical_gradient(struct('f', {loss_W, self.B2, 'B2', net}));
            grads = {grad_W1 grad_B1 grad_W2 grad_B2};
        end
        
        % Analytical Gradient ------------------------------------------
        function grads = analytical_gradient(self, X, T, param)
            % forward propagation
            self.loss(X, T, param);
            
            
            dout = 1;
            [doutAE, doutGP] = self.AddLayer.backward(dout);
            
            % AE side
            doutAE = self.SquareError.backward(doutAE);
            doutAE = self.Sigmoid2.backward(doutAE);
            doutAE = self.Affine2.backward(doutAE);
            
            % GP side 
            doutGP = self.MullLayler.backward(doutGP);
            doutGP = self.LogLikelihood.backward(doutGP);
            
            % combine AE and GP backprop
            dout = self.Sigmoid1.backward(doutAE, doutGP);
            dout = self.Affine1.backward(dout);
            
            % set gradient
            grad_W1 = self.Affine1.dW;
            grad_B1 = self.Affine1.dB;
            grad_W2 = self.Affine2.dW;
            grad_B2 = self.Affine2.dB;
            grad_GPparam = self.LogLikelihood.dparam2;
            
            grads = {grad_W1 grad_B1 grad_W2 grad_B2 grad_GPparam};
            
            
        end
        
        % Set Parameters -----------------------------------------------
        function self = set_param(self, param, pname)
            if      pname == 'W1';  self.W1 = param;
            elseif  pname == 'B1';  self.B1 = param;
            elseif  pname == 'W2';  self.W2 = param;
            elseif  pname == 'B2';  self.B2 = param;
            end
        end
    end
end

    