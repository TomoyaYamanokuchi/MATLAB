classdef GPLVM_Main < handle
    properties
        x_train, t_train;
        regression;
        kernel;
    end
    methods 
    % set data 
    function self = GPLVM_Main(x, t)
        % set data
        self.x_train = x;
        self.t_train = t;    
    end
        
    
    % set data and GPregression 
    function [f, g] = GaussianProcess(self, param)
        % set kernel param --> Gaussiankernel[É∆0, É∆1]
        self.kernel = GPLVM_Kernel_ver1([param(1), param(2)]); 
%         self.kernel = GPLVM_Kernel([param(1), param(2)]); 
        
        % set kernel and precision É¿ of covariance matrix --> GPRegression[kernel, É¿]
        self.regression = GPLVM_Regression_ver1(self.kernel, param(3));
%         self.regression = GPLVM_Regression(self.kernel, param(3));

        % calculate gradient of log-likelihoot function
%         g = cell(1, 4);
        g = self.regression.gradient_log_likelihood(self.x_train, self.t_train);
        g = 0.5*[-g{1} -g{2} -g{3} -g{4}] % <====!!!
        

        
        % calculate log-likelihoot function
        f = - self.regression.log_likelihoot(self.t_train)
        
                hbhb = kmkm
    end
    
    
    % show predict distribution after optimization 
    function y = predict(self, x_test) 
        [y, y_var] = self.regression.predict_dist(x_test);
    end
    
    
    % set parameter
    function set(self, param) 
        % set kernel param --> Gaussiankernel[É∆0, É∆1] 
        self.kernel = GPLVM_Kernel([param(1), param(2)]); 
        % set kernel and precision É¿ of covariance matrix --> GPRegression[kernel, É¿]
        self.regression = GPLVM_Regression(self.kernel, param(3));
        self.regression.fit(self.x_train, self.t_train);
    end
    

    end
end

