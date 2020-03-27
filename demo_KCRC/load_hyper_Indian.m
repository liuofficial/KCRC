function [Train, Test] = load_hyper_Indian(n, m)
[img, img_gt, nClass, rows, cols, bands] = load_datas(1);
[train_idx, test_idx] = load_train_test(1, 2, n, m);
[Train, Test] = set_train_test(train_idx, test_idx, img, img_gt, 1);
end