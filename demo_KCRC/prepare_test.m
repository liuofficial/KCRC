function [P] = prepare_test(traindat, testdat, kersize, w, scales, funs, paras)
trainsize = size(traindat,2); testsize = size(testdat,2);
P = zeros(trainsize,testsize);
for i = 1 : kersize
    fun = funs{i};
    P = P + w(i) / scales(i) * fun(traindat,testdat,paras(i));
end
end