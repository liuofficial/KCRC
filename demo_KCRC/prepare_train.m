function [Qs,scales] = prepare_train(traindat, kersize, funs, paras)
trainsize = size(traindat,2);
Qs = zeros(trainsize,trainsize,kersize);
scales = zeros(1,kersize);
for i = 1 : kersize
    fun = funs{i};
    tmp = fun(traindat,traindat,paras(i));
    scales(i) = mean(diag(tmp));
    Qs(:,:,i) = tmp / scales(i);
end
end