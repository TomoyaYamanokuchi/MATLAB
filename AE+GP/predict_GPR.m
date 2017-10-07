function predict_GPR(optimparam, x_origin)

x_train = [1:9 0];
x_test = [1:9 0];
gpr = zeros(size(x_origin));
n = length(x_origin);


    tic;

for j=1:length(x_test)
    for i=1:n
        gp = GP_class_image(x_train', x_origin(:,i));
        gpr(j, i) =  gp.predict(x_test(j), optimparam);
    end
end

    toc;
    
    

% show figure =============================================================

a1 = 1;
a2 = 3;
b = 4;

figure;
os_size = sqrt(n);
width = 4;

for i=1:length(x_test)
if i<6 
    subplot(5, width, a1 + b*(i-1));  
    image(reshape(x_origin(i,:), os_size, os_size)*255);
    title('Original');
    
    subplot(5,width, a1 + b*(i-1)+1);  
    image(reshape(gpr(i,:), os_size, os_size)*255);
    title('GPR');
else
    j = i-5;
    subplot(5,width, a2 + b*(j-1));  
    image(reshape(x_origin(i,:), os_size, os_size)*255);
    title('Original');
    
    subplot(5,width, a2 + b*(j-1)+1);  
    image(reshape(gpr(i,:), os_size, os_size)*255);
    title('GPR');
end
end


    
    
end