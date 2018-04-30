function X_filtered = lp_filter(X)

X_r = X(:, :, :, 1);
X_g = X(:, :, :, 2);
X_b = X(:, :, :, 3);
Xa = X(:, :, :, 4);

sigma = 0.5;

% Convert to double
Xa_double = double(Xa) / 255;
X_r_double = double(X_r) / 255;
X_g_double = double(X_g) / 255;
X_b_double = double(X_b) / 255;

% Apply filter
X_a_filtered = imgaussfilt3(Xa_double, sigma);
X_r_filtered = imgaussfilt3(X_r_double, sigma) ./ X_a_filtered;
X_g_filtered = imgaussfilt3(X_g_double, sigma) ./ X_a_filtered;
X_b_filtered = imgaussfilt3(X_b_double, sigma) ./ X_a_filtered;

% Convert to uint
X_r_filtered = uint8(X_r_filtered * 255);
X_g_filtered = uint8(X_g_filtered * 255);
X_b_filtered = uint8(X_b_filtered * 255);

% Reset colors of unoccupied voxels
unoccupied = find(~Xa);
X_r_filtered(unoccupied) = zeros(size(unoccupied));
X_g_filtered(unoccupied) = zeros(size(unoccupied));
X_b_filtered(unoccupied) = zeros(size(unoccupied));

X_filtered = cat(4, X_r_filtered, X_g_filtered, X_b_filtered, Xa);

end