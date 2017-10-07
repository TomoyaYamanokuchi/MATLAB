classdef LogLikelihood < handle
    properties
        gp, param, dparam1, dparam2;
        low;
    end
    methods
        function self = LogLikelihood()
        end
        
        function f = forward(self, lowdim, param)
            m = size(lowdim, 2);
            
            
            x = [1:9 0];
%             x = [2:9 0];
            self.low = lowdim;
            f = 0;
            
            self.dparam1 = [0 0 0];
            for i=1:m
                self.gp = GP_class_image(x', lowdim(:,i));
                [f_temp, dparam1_temp] = self.gp.GaussianProcess(param);
                f = f + f_temp;
                self.dparam1 = self.dparam1 + dparam1_temp;
            end
        end
        
        function dx = backward(self, dout)
            dx = zeros(size(self.low));
            
            
            for i=1:size(self.low,2)
                dx(:,i) = -( - self.gp.regression.precision * self.low(:,i)) * dout;
            end
            
            
            self.dparam2 = dout*self.dparam1;
        end
    end
end