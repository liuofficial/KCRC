function [f, g] = loocv_rbf_app(x, dist, T)
% 2018-09-12
gam = x(1); lam = x(2);
Q = exp(-dist * gam);
G = inv(Q + lam*eye(size(Q)));
P = T - bsxfun(@times, T*G, 1./diag(G)');
C = size(T,1);
P = C * P;
P = bsxfun(@plus, P, -max(P) + 1);
P = exp(P);
P = bsxfun(@times, P, 1./sum(P));
%f = 0.5*norm(P.*T-T,'fro')^2;
f = sum(sum(-P.*T));

if nargout > 1 % gradient required
    iG = 1./diag(G);
    if find(iG<eps)
        warning('Division Loss');
        iG = max(iG, eps);
    end
    dydP = P.*T - T;
    %dydP = -bsxfun(@times, P, sum(dydP.^2)+sum(dydP)) + dydP .* dydP + dydP;
    dydP = P - P.*T + bsxfun(@times, P, sum(dydP));
    dydP = dydP * C;
    dydG = bsxfun(@times, T'*dydP, iG');
    dydG = - dydG + diag(diag(bsxfun(@times, G*dydG, iG')));
    dydQ = - G*dydG*G;
    dydls = trace(dydQ');
    dydgs = trace(dydQ' * (-Q .* dist));
    g = [dydgs; dydls];
end
end