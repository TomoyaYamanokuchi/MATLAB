function y = softmax(a)
C = max(a);
y = exp(a-C) ./ sum(exp(a-C)) ;
end

