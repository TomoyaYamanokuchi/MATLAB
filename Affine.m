classdef Affine < handle
    properties
        W;
        B;
        X;
        dW;
        dB;
    end
    methods
        function self = Affine(W, B)
            self.W = W;
            self.B = B;
            self.X = [];
            self.dW = [];
            self.dB = [];
        end

        
        function out = forward(self, x, w, b)
            self.X = x;
%             disp('---X-----------------')
%             disp(self.X)
            self.W = w;
%             disp('---W-----------------')
%             disp(self.W)
            self.B = b;
%             disp('---B-----------------')
%             disp(self.B)
           
            out = x*self.W + self.B;
        end
        
        function dx = backward(self, dout) 
            dx = dout * self.W';
            self.dW = self.X' * dout;
            
            % sum of column
            self.dB = sum(dout, 1);
        end
    end
end