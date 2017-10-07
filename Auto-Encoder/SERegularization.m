classdef SERegularization < handle
    properties
        err, y, t
        W1, B1, W2, B2;
    end
    methods
        function self = SERegularization()
            self.err = [];
            self.W1 = [];
            self.B1 = [];
            self.W2 = [];
            self.B2 = [];
        end
        
        function L = forward(self, Y, T, obj)
            self.y = Y;
            self.t = T;
            self.W1 = obj.W1;
            self.B1 = obj.B1;
            self.W2 = obj.W2;
            self.B2 = obj.B2;
            
            L = (0.5*sum( sum((Y-T).^2, 2)))...
                + 0.5*((self.W1'*self.W1)+(self.B1'*self.B1)+(self.W2'*self.W2)+(self.B2'*self.B2));
        end
        
        function dout = backward(self, dout)
            dout = ((self.y - self.t) + (self.W1 + self.B1 + self.W2 + self.B2)).* dout;
        end
    end
end