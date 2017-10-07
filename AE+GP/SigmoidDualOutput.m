classdef SigmoidDualOutput < handle
    properties
        out
    end
    methods
        function self = SigmoidDualOutput()
            self.out = [];
        end
        
        function out = forward(self, x)
            out = 1 ./ (1 + exp(-x));
            self.out = out ;
           
        end
        
        function dx = backward(self, doutAE, doutGP) 
            dx = (doutAE + doutGP) .* (1.0 - self.out) .* self.out;
        end
    end
end