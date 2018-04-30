function [X_subsampled, meta] = filter_and_sample(highres_filename, output_res)
[X, meta] = nrrdread(highres_filename);
X = permute(X, [3, 4, 2, 1]);

Xrgb = X(:,:,:,1:3);
Xa = X(:, :, :, 4);

X_filtered = lp_filter(X);
X_r_filtered = X_filtered(:, :, :, 1);
X_g_filtered = X_filtered(:, :, :, 2);
X_b_filtered = X_filtered(:, :, :, 3);

Xrgb_filtered = cat(4, X_r_filtered, X_g_filtered, X_b_filtered);

X_highres_filtered = cat(4, Xrgb_filtered, Xa);

X_subsampled = subsample(X_highres_filtered, output_res);
end