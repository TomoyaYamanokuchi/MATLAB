function [c, ceq, gc, gceq] = MPP_3_OP_confungrad(x)
c = [];
ceq = [];
if nargout > 2
    gc = [];
    gceq = [];
end