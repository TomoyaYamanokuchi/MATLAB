classdef ReLu < handle
    properties
        mask
    end
    methods
        function self = ReLu()
            % do nothing
        end
        
        function out = forward(self, x)
            x(x <= 0) = 0;
            self.mask = x;
            out = x; 
        end
        
        function dx = backward(self, dout) 
            dout(self.mask == 0) = 0;
            dx = dout;
        end
    end
end