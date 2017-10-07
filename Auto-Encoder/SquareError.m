classdef SquareError < handle
    properties
        err, y, t
    end
    methods
        function self = SquareError()
            self.err = [];
        end
        
        function L = forward(self, Y, T)
            self.y = Y;
            self.t = T;
            L = (0.5*sum( sum((Y-T).^2, 2)));
        end
        
        function dout = backward(self, dout)
            dout = (self.y - self.t) .* dout;
        end
    end
end