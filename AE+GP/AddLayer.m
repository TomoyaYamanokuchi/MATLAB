classdef AddLayer < handle
    properties
    end
    methods
        function self = AddLayer()
            % do nothing
        end
        
        function out = forward(~, x, y)
           out = x + y; 
        end
        
        function [dx, dy] = backward(~, dout)
           dx = dout*1;
           dy = dout*1;
        end
    end
end