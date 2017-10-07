function Q = func_myp53(P, pmin, pmax)
P(P<pmin) = pmin;
P(P>pmax) = pmax;
Q = P;
end