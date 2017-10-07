function bout = Batch_Normalization(x)

m = size(x, 2)
mub = sum(x') ./ m
func_mub = mean(x, 2)
size(mub)

varb = sum(((x - mub').^2)') ./ m
func_var = var((x-func_mub).^2, 0, 2)

size(varb)

e = 10e-7;

bout = (x - mub') ./ sqrt(varb' + e)


% ans_mean = mean(bout, 2)
% mean(bout(1, :), 2)
% mean(bout(2, :), 2)


ans_var = var(bout, 0, 2)
var(bout(1, :))
var(bout(2, :))

