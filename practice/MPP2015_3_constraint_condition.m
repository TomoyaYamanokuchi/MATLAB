function [A, b, Aeq, beq, lb, ub, nonlcon] = MPP2015_3_constraint_condition
A = [];
b = [];
Aeq = [];
beq = [];
% vector notation is used for range restriction
lb = [-3 -2];
ub = [3 2];
nonlcon = [];
%opts = optimoptions(@fmincon,'Display','iter','Algorithm','interior-point');
