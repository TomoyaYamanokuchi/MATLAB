function fronorm = AEGP_show(optim_param, x_origin)

m = optim_param{3};
gamma = optim_param{4};

% get low-dim using optimization parameter W and B---------------
net = set_GPNet(optim_param{1,1}, x_origin, m, gamma);
[~, low] = net.predict(x_origin);


% inference for GPR -----------------------------------------------
x_test_gp = [1:9 0];
x_size = size(x_test_gp, 2);
gpr = zeros(size(low));

% class set 
gp = cell(1, m);
for i=1:m
    gp{i} = GP_class_image(x_test_gp', low(:,i));
    gp{i}.set(optim_param{2});
end

tic;
% predict
for j=1:x_size
    for i=1:m
        gpr(j, i) =  gp{i}.predict(x_test_gp(j));
    end
end



% gpr decode -----------------------------------------------------
gpr_decode = net.decode(gpr);

toc;

% squared error --------------------------------------------------
fronorm =  norm(x_origin - gpr_decode, 'fro');


% show graph ------------------------------------------------------
a1 = 1;
a2 = 5;
b = 8;

os_resize = sqrt(m); % One Side of a resize image
os_size = sqrt(size(x_origin, 2));

figure(1);
length = 5;
width = 8;
for i=1:10 %------------------
if i<6 
    subplot(length, width, a1 + b*(i-1));  
    image(reshape(x_origin(i,:), os_size, os_size)*255);
    title('Original');
    
    subplot(length,width, a1 + b*(i-1)+1);  
    image(reshape(gpr_decode(i,:), os_size, os_size)*255);
    title('Restored');
    
    subplot(length,width, a1 + b*(i-1)+2);  
    image(reshape(low(i,:), os_resize, os_resize)*255);
    title('Low-Dim');
    
    subplot(length,width, a1 + b*(i-1)+3);  
    image(reshape(gpr(i,:), os_resize, os_resize)*255);
    title('GPR Low-Dim');
else
    j = i-5;
    subplot(length,width, a2 + b*(j-1));  
    image(reshape(x_origin(i,:), os_size, os_size)*255);
    title('Original');
    
    subplot(length,width, a2 + b*(j-1)+1);  
    image(reshape(gpr_decode(i,:), os_size, os_size)*255);
    title('Restored');
    
    subplot(length,width, a2 + b*(j-1)+2);  
    image(reshape(low(i,:), os_resize, os_resize)*255);
    title('Low-Dim');
    
    subplot(length,width, a2 + b*(j-1)+3);  
    image(reshape(gpr(i,:), os_resize, os_resize)*255);
    title('GPR Low-Dim');
end

end

