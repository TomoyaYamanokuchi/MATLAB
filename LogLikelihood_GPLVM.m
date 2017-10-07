classdef LogLikelihood_GPLVM < handle
    properties
        gplvm, param_num, dparam1, dparam_param, dparam_input;
        low;
    end
    methods
        function self = LogLikelihood_GPLVM()
        end
        
        function f = forward(self, lowdim, param, x)
            m = size(lowdim, 2);
            self.param_num = length(param);
            self.low = lowdim;
            f = 0;
            self.dparam1 = zeros(1, length(param)+(size(x,1)*size(x,2)));
            for i=1:m
                self.gplvm = GPLVM_Main(x, lowdim(:,i));
                [f_temp, dparam1_temp] = self.gplvm.GaussianProcess(param);
                f = f + f_temp;
                self.dparam1 = self.dparam1 + dparam1_temp;
            end
        end
        
        function dx = backward(self, dout)
            dx = zeros(size(self.low));
            
            for i=1:size(self.low,2)
                dx(:,i) = -( - self.gplvm.regression.precision * self.low(:,i)) * dout;
            end
            
            self.dparam_param = dout*self.dparam1(1:self.param_num);
            self.dparam_input = dout*self.dparam1(self.param_num+1:end);
        end
    end
end