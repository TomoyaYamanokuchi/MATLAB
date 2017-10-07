function graph_script3(data) 

x_train = data{2};
x_rest = data{3};
low = data{4};


a1 = 1;
a2 = 4;
b = 6;

figure(1);
length = 5;
width = 6;

os_size = sqrt(size(x_train, 2));
os_resize = sqrt(size(low, 2));

for i=1:10 %------------------
if i<6 
    subplot(length, width, a1 + b*(i-1));  
    image(reshape(x_train(i,:), os_size, os_size)*255);
    title('Original Image');
    
    subplot(length,width, a1 + b*(i-1)+1);  
    image(24, 24, reshape(x_rest(i,:), os_size, os_size)*255);
    title('Restored Image');
    
    subplot(length,width, a1 + b*(i-1)+2);  
    image(reshape(low(i,:), os_resize, os_resize)*255);
    title('Restored Image');
else
    j = i-5;
    subplot(length,width, a2 + b*(j-1));  
    image(reshape(x_train(i,:), os_size, os_size)*255);
    title('Original Image');
    
    subplot(length,width, a2 + b*(j-1)+1);  
    image(24, 24, reshape(x_rest(i,:), os_size, os_size)*255);
    title('Restored Image');
    
    subplot(length,width, a2 + b*(j-1)+2);  
    image(reshape(low(i,:), os_resize, os_resize)*255);
    title('Restored Image');
end
end 



end