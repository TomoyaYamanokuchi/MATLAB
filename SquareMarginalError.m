classdef SquareMarginalError < handle
    properties
        out
    end
    methods
        function self = SquareMarginalError()
            self.out = [];
        end
        
        function out = forward(self, x)
            out = 1 ./ (1 + exp(-x));
            self.out = out ;
           
        end
        
        function dx = backward(self, dout) 
            dx = dout .* (1.0 - self.out) .* self.out;
        end
    end
end