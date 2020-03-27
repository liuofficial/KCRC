function acc = comp_pkcrc(Train, Test, gam, lam, T)
sig = sqrt(0.5 / gam);
P = RBF(Train.dat, Test.dat, sig, 5000);
Q = RBF(Train.dat, Train.dat, sig, 5000);
TS = pkcrc_solve(Q, P, T, lam);
[~, pred] = max(TS);
acc = class_eval(pred, Test.lab);
end