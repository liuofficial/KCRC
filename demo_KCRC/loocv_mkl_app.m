function [f, g] = loocv_mkl_app(x, Qs, T)
lam = 1e-3;
w = reshape(x,1,1,numel(x));
Q = sum(bsxfun(@times,Qs,w), 3);
G = inv(Q + lam*eye(size(Q)));
P = T - bsxfun(@times, T*G, 1./diag(G)');
C = size(T,1);
P = C * P;
P = bsxfun(@plus, P, -max(P) + 1);
P = exp(P);
P = bsxfun(@times, P, 1./sum(P));
f = sum(sum(-P.*T));

if nargout > 1 % gradient required
    dydP = P.*T - T;
    dydP = P - P.*T + bsxfun(@times, P, sum(dydP));
    dydP = dydP * C;
    dydG = bsxfun(@times, T'*dydP, 1./diag(G)');
    dydG = - dydG + diag(diag(bsxfun(@times, G*dydG, 1./diag(G)')));
    dydQ = - G*dydG*G;
    g = zeros(length(w),1);
    for i = 1 : length(w)
        g(i) = trace(dydQ' * Qs(:,:,i));
    end
end
end