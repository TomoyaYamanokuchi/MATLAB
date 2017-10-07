classdef GP_class_image < handle
    properties
%         x_real,  t_real;
        x_train, t_train;
        x_test;
        regression;
        kernel;
    end
    methods 
    % set data 
    function self = GP_class_image(data)
        % set data
        self.x_train = data{1};
        self.t_train = data{2};
        self.x_test = data{3};
%         self.x_real = data{4};
%         self.t_real = data{5};
    end
        
    
    % set data and GPregression 
    function [f, g] = GaussianProcess(self, param)
        % set kernel param --> Gaussiankernel[É∆0, É∆1] 
        self.kernel = GaussianKernel([param(1), param(2)]); 
        
        % set kernel and precision É¿ of covariance matrix --> GPRegression[kernel, É¿]
        self.regression = GPRegression(self.kernel, param(3));
        
        % calculate gradient of log-likelihoot function
        g = cell(1, 3);
        g = self.regression.gradient_log_likelihood(self.x_train, self.t_train);
        g = [-g{1};
             -g{2};
             -g{3}];
        size(g)
        % calculate log-likelihoot function
        f = -self.regression.log_likelihoot(self.t_train);
        
    end
    
    
    % show predict distribution after optimization 
    function show(self) 
        [y, y_var] = self.regression.predict_dist(self.x_test);
    
        figure;
        X = [self.x_test, fliplr(self.x_test)];
        Y = [y'-2*sqrt(y_var'), fliplr(y'+2*sqrt(y_var'))];
        gray = [0.85 0.85 0.85];
        fill(X, Y, gray);                       hold on;
        
        plot(self.x_test, y, '-r');             hold on;
        plot(self.x_real, self.t_real, '-b');   hold on; 
        plot(self.x_train, self.t_train, 'xk'); hold on;
        
        ylim([-1.5 1.5])
        legend('95%confidence-band', 'predict_mean', 'sin(2ÉŒx)',  'observation');
    end
    end
end

