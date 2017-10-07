classdef TwoLayerNet < handle
    properties
       ;
       y;
    end
    methods
        function self = TwoLayerNet()
            self. = [];
            self.y = [];
        end
        
        function out = forward(self, x, y)
            self.x = x;
            self.y = y;
            out = x * y;
        end
        
        function [dx, dy] = backward(self, dout)
            dx = dout*self.y;
            dy = dout*self.x;
        end
    end
end
    