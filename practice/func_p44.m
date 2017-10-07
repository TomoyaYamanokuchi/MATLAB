function v = func_p44(D)
v = D(1, 1);
for j = 1:size(D, 2)
    for i = 1:size(D, 1)
        if v < D(i, j)
            v = D(i, j);
        end
    end
end
end
