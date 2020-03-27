function [loss] = grid_search(gams, lams, fun)
gam_size = length(gams); lam_size = length(lams);
loss.vals = zeros(gam_size*lam_size,3);
k = 0;
for i = 1 : gam_size
    g = gams(i);
    for j = 1 : lam_size
        l = lams(j);
        k = k + 1;
        oa = fun([g; l]);
        loss.vals(k,:) = [g l oa];
    end
end
[val,k] = min(loss.vals(:,3));
loss.gam = loss.vals(k, 1); loss.lam = loss.vals(k, 2);
loss.val = val;
end