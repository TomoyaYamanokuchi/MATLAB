function data = data_sin()

        % real function
        function [x, y] = real_function()
            x = 0:0.01:1;
            y = sin(2*pi*x);
        end
        [x_real, t_real] = real_function();
        
 
        % generate training datas
        function [x, y] = train_generate()
             std = 0.1; b = 0.3; c = 0.7;
             
             % range of x is from b to c
             x = (c-b).*rand(1, 10)+b;
%            b = 0.7; c = 0.9;
%            x = [x (c-b).*rand(1, 10)+b];

             y = sin(2*pi*x) + randn(1, length(x)).*std;
        end
        [x_train, t_train] = train_generate();
        
        
        x_test = 0:0.01:1;
        
        data = { x_train' t_train', x_test' x_real t_real };

end