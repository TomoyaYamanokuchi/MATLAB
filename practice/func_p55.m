function R = func_p55(S)
N = size(S, 2);
R = zeros(N);
for i = 1:N
    for j = 1:N
        R(i,j) = norm(S(:,i) - S(:,j));
    end
end
end