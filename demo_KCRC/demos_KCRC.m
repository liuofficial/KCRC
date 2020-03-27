% Learning Multiple Parameters for Kernel Collaborative Representation Classification, TNNLS, 2020
clear; clc;

% data: Indian Pines (Hyperspectral Remote Sensing Image)
nt = 3; it = 1;
[Train, Test] = load_hyper_Indian(nt, it);

T = get_coef_lab(Train.lab);

% single kernel learning
disp('begin: single kernel learning');
dist = ker_euclid(Train.dat, Train.dat);
disp('cross validation...');
gams = 2.^(-20:0.5:20); lams = 10.^(-6:0.5:6);
tic;
[skl_cv] = grid_search(gams, lams, @(x)loocv_rbf_app(x,dist,T)); 
toc;
skl_cv_acc = comp_pkcrc(Train, Test, skl_cv.gam, skl_cv.lam, T);
disp(skl_cv); disp(skl_cv_acc);
disp('leave one out learing...');
tic;
[skl] = opt_learn(@(x)loocv_rbf_app(x,dist,T));
toc;
skl_acc = comp_pkcrc(Train, Test, skl.gam, skl.lam, T);
disp(skl); disp(skl_acc);

% multiple learning learning
disp('begin: multiple kernel learning');
% four kinds of kernels
f_rbf = @(d1,d2,x) RBF(d1,d2,x,5000);
f_poly = @(d1,d2,x) POLY(d1,d2,x,5000);
f_cos = @(d1,d2,x) COSINE(d1,d2,5000);
f_lin = @(d1,d2,x) LIN(d1,d2,5000);
lam = 1e-3;
gams = 2.^(-15:2:5); ds = 2 : 4;
gams_size = length(gams); ds_size = length(ds);
sigs = sqrt(0.5 ./ gams);
ker_size = gams_size + ds_size + 2;
funs = cell(1,ker_size); paras = zeros(1,ker_size);
for i = 1 : gams_size
    funs{i} = f_rbf;
    paras(i) = sigs(i);
end
for i = 1 : ds_size
    funs{i+gams_size} = f_poly;
    paras(i+gams_size) = ds(i);
end
funs{end-1} = f_cos;
funs{end} = f_lin;
% begin
[Qs,scales] = prepare_train(Train.dat, ker_size, funs, paras);
[mkl] = opt_learn_mk(@(x)loocv_mkl_app(x,Qs,T), ker_size);
w = reshape(mkl.mu,1,1,numel(mkl.mu));
Q = sum(bsxfun(@times,Qs,w), 3);
P = prepare_test(Train.dat, Test.dat, ker_size, w, scales, funs, paras);
TS = T / (Q + lam*eye(size(Q))) * P;
[~, pred] = max(TS);
mkl_acc = class_eval(pred, Test.lab);
disp(mkl); disp(mkl_acc);