function Mhat = func_p54(M)
N = size(M, 2);
mbar = mean(M, 2);
Mhat = zeros(size(M));
for i = 1:N
    Mhat(:, i) = M(:, i) - mbar;
end
