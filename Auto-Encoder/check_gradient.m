function grad = check_gradient(X, l)
import TwoLayerNet;
network = TwoLayerNet(X, l);




grad_numerical = network.numerical_gradient(X, X, network);
grad_analytical  = network.analytical_gradient(X, X);


disp('grad_numerical=======================')
disp('grad_W1_nume = ')
disp(grad_numerical{1})
disp('grad_B1_nume = ')
disp(grad_numerical{2})
disp('grad_W2_nume = ')
disp(grad_numerical{3})
disp('grad_B2_nume = ')
disp(grad_numerical{4})

disp('grad_analytical=====================')
disp('grad_W1 = ')
disp(grad_analytical{1})
disp('grad_B1 = ')
disp(grad_analytical{2})
disp('grad_W2 = ')
disp(grad_analytical{3})
disp('grad_B2 = ')
disp(grad_analytical{4})

% disp('grad_formula===========================')
% disp('grad_W1 = ')
% disp(grads_formula{1})
% disp('grad_B1 = ')
% disp(grads_formula{2})
% disp('grad_W2 = ')
% disp(grads_formula{3})
% disp('grad_B2 = ')
% disp(grads_formula{4})


W1_error = abs(grad_numerical{1} - grad_analytical{1});
B1_error = abs(grad_numerical{2} - grad_analytical{2});
W2_error = abs(grad_numerical{3} - grad_analytical{3});
B2_error = abs(grad_numerical{4} - grad_analytical{4});

ave_W1_error = mean(mean((W1_error), 2))
ave_B1_error = mean(mean(B1_error))
ave_W2_error = mean(mean((W2_error), 2))
ave_B2_error = mean(mean(B2_error))



end