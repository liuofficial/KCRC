function K = LIN(X, X2, block)
% linear kernel
% 2012_09_20
n1 = size(X,2);  n2 = size(X2,2);
if (n1 < block) && (n2 < block),
    K = X' * X2;
else
    K = zeros(n1, n2);
    n1_block = ceil(n1/block); n2_block = ceil(n2/block);
    for i = 1 : n1_block
        n1_idx = (i-1)*block+1 : min(i*block, n1);
        for j = 1 : n2_block
            n2_idx = (j-1)*block+1 : min(j*block, n2);
            K(n1_idx, n2_idx) = X(:,n1_idx)' * X2(:,n2_idx);
        end
    end
end
end