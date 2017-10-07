function [f, g] = autofunctionGP(x, t, param)

import GP_class_image;
gp = GP_class_image(x, t);

[f, g] = gp.GaussianProcess(param);

end