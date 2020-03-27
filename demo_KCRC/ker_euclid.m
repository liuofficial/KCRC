function dist = ker_euclid(A, X)
nA = sum(A.^2); nX = sum(X.^2);
[mX,mA] = meshgrid(nX,nA);
dist = mA-2*A'*X+mX;
end