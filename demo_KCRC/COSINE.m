function K = COSINE(X, X2, block)
% cosine kernel
% 2013_10_13
n1 = size(X,2);  n2 = size(X2,2);
if (n1 < block) && (n2 < block),
    X = bsxfun(@times, X, 1./sqrt(sum(X.^2))); X2 = bsxfun(@times, X2, 1./sqrt(sum(X2.^2)));
    K = X' * X2;
else
    K = zeros(n1, n2);
    n1_block = ceil(n1/block); n2_block = ceil(n2/block);
    for i = 1 : n1_block
        n1_idx = (i-1)*block+1 : min(i*block, n1);
        for j = 1 : n2_block
            n2_idx = (j-1)*block+1 : min(j*block, n2);
            Xtmp = X(:,n1_idx); X2tmp = X2(:,n2_idx);
            Xtmp = bsxfun(@times, Xtmp, 1./sqrt(sum(Xtmp.^2))); X2tmp = bsxfun(@times, X2tmp, 1./sqrt(sum(X2tmp.^2)));
            K(n1_idx, n2_idx) = Xtmp' * X2tmp;
        end
    end
end
end
