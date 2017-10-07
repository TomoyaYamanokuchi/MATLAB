function v = func_p43(D)
v = D(1, 1);
for i = 1:size(D, 1)
    for j = 1:size(D, 2)
        if v < D(i, j)
            v = D(i, j);
        end
    end
end
end