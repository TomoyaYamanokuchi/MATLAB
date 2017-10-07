classdef TwoLayerNet < handle
    properties
       W1, B1, W2, B2, W3, B3, W4, B4, W5, B5, W6, B6;
       Affine1, Affine2, Affine3, Affine4, Affine5, Affine6;
       Sigmoid1, Sigmoid2;
       ReLu1, ReLu2, ReLu3, ReLu4,ReLu5, ReLu6,ReLu7;
       SquareError;
       SERegularization;
    end
    methods
        function [self, param] = TwoLayerNet(X, l)
            n = size(X, 2); % dimension of input data
            m = n;          % dimension of output data
            
            % format for fmin optimization function
            % generation of Weights and Biases
            self.W1 = randn(n, l) ./ sqrt(n); 
            self.B1 = zeros(1, l);
            self.W2 = randn(l, m) ./ sqrt(l);
            self.B2 = zeros(1, m);
            
             
            W1 = reshape(self.W1, 1, n*l);
            W2 = reshape(self.W2, 1, l*m);
            
            param = horzcat(W1, self.B1, W2, self.B2);
            
            
            % generation of Layer ===================================
            %
            % Activation Function is Sigmoid ************************
            
            self.Affine1 = Affine(self.W1, self.B1);
            self.Sigmoid1 = Sigmoid();
            self.Affine2 = Affine(self.W2, self.B2);
            self.Sigmoid2 = Sigmoid();
            self.SquareError = SquareError();
%             self.SERegularization = SERegularization();
            %
%             % Activation Function is ReLu ****************************
%             self.Affine1 = Affine(self.W1, self.B1);
%             self.ReLu1 = ReLu();
%             self.Affine2 = Affine(self.W2, self.B2);
%             self.ReLu2 = ReLu();
%             self.MSE = MeanSquaredError();
%             
        end
        
        
        % forward propagation ----------------------------------------
        function y = predict(self, x)
            
            % Sigmoid
            y = self.Affine1.forward(x, self.W1, self.B1);
            y = self.Sigmoid1.forward(y);
            y = self.Affine2.forward(y, self.W2, self.B2);
            y = self.Sigmoid2.forward(y);

%             % Relu 
%             y = self.Affine1.forward(x, self.W1, self.B1);
%             y = self.ReLu1.forward(y);
%             y = self.Affine2.forward(y, self.W2, self.B2);
%             y = self.ReLu2.forward(y);

        end
        
        
        % calculate loss function -------------------------------------
        function L = loss(self, X, T)
            Y = self.predict(X);
            L = self.SquareError.forward(Y, T);
%             L = self.SERegularization.forward(Y, T, self);
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
        function grads = analytical_gradient(self, X, T)
            % forward propagation
            self.loss(X, T);
            
            % Sigmoid back propagation
            dout = 1;
            dout = self.SquareError.backward(dout);
%             dout = self.SERegularization.backward(dout);
            dout = self.Sigmoid2.backward(dout);
            dout = self.Affine2.backward(dout);
            dout = self.Sigmoid1.backward(dout);
            dout = self.Affine1.backward(dout);
            
%             % ReLu back propagation
%             dout = 1;
%             dout = self.MSE.backward(dout);
%             dout = self.ReLu2.backward(dout);
%             dout = self.Affine2.backward(dout);
%             dout = self.ReLu1.backward(dout);
%             dout = self.Affine1.backward(dout);
            
            
            % set gradient
            grad_W1 = self.Affine1.dW;
            grad_B1 = self.Affine1.dB;
            grad_W2 = self.Affine2.dW;
            grad_B2 = self.Affine2.dB;
            
            grads = {grad_W1 grad_B1 grad_W2 grad_B2};
            
            
        end
        
        % Set Parameters -----------------------------------------------
        function self = set_param(self, param, pname)
            if      pname == 'W1';  self.W1 = param;
            elseif  pname == 'B1';  self.B1 = param;
            elseif  pname == 'W2';  self.W2 = param;
            elseif  pname == 'B2';  self.B2 = param;
%             elseif  pname == 'W3';  self.W3 = param;
%             elseif  pname == 'B3';  self.B3 = param;
%             elseif  pname == 'W4';  self.W4 = param;
%             elseif  pname == 'B4';  self.B4 = param;
%             elseif  pname == 'W5';  self.W5 = param;
%             elseif  pname == 'B5';  self.B5 = param;
%             elseif  pname == 'W6';  self.W6 = param;
%             elseif  pname == 'B6';  self.B6 = param;
            end
        end
    end
end

    
    
    
    
    
    