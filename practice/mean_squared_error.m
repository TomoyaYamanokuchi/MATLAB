function L = mean_squared_error(Y, T)
N = size(Y, 1);
L = (1/2)*(1/N)*sum(sum(((Y-T).^2)'));
end