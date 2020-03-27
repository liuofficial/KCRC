function TS = pkcrc_solve(Q, P, T, lam)
TS = T / (Q + lam*eye(size(Q))) * P;
end