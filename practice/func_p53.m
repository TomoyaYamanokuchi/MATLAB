function Q = func_p53(P, pmin, pmax)
% Q(matrix), P(matrix), pmin/pmax(scalar)
Q = zeros(size(P));
for i = 1:size(P, 1)
    for j = 1:size(P, 2)
        if P(i,j) < pmin
            Q(i,j) = pmin;
        elseif P(i,j) > pmax
            Q(i,j) = pmax;
        else 
            Q(i,j) = P(i,j);
        end
    end
end
end