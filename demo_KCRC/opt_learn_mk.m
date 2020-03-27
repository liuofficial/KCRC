function [dat] = opt_learn_mk(fun, kersize, verb)
if nargin < 3
    verb = 'off';
end
options = optimoptions('fmincon',...
    'Algorithm','interior-point',...
    'Display',verb,...
    'SpecifyObjectiveGradient',true);
problem.objective  = @(x) fun(x);
problem.x0 = ones(1,kersize) / kersize;
problem.Aeq = ones(1,kersize);
problem.beq = 1;
problem.lb = zeros(kersize,1);
problem.ub = ones(kersize,1);
problem.solver = 'fmincon';
problem.options = options;
[x, fval, exitflag, output, ~, grad] = fmincon(problem);
dat = [];
dat.mu = x;
dat.val = fval;
dat.fct = output.funcCount;
end