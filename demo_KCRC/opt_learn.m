function [loss] = opt_learn(fun, verb)
if nargin < 2
    verb = 'off';
end
lb = [2^-20; 10^-6]; ub = [2^20; 10^6];
options = optimoptions('fmincon',...
    'Algorithm','interior-point',...
    'Display',verb,...
    'SpecifyObjectiveGradient',true);
x0 = [1; 1];
loss.val0 = fun(x0);
[x, fval, ~, output, ~, grad] = fmincon(@(x) fun(x), ...
    x0, [], [], [], [], lb, ub, [], options);
loss.gam = x(1); loss.lam = x(2);
loss.val = fval;
loss.fct = output.funcCount;
end