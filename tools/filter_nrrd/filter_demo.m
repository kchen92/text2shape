addpath('../matlab')

%% Settings
nrrd_dir = '/ssd105/kchen/nrrd_32_solid';
highres_nrrd_dir = '/ssd105/kchen/nrrd_256_solid';
output_res = 32;

%% Model IDs to visualize
model_ids = { ...
    '9930f9ec8e2e94b4febad4f49b26ec52'; ...
    '73abab8429d1694c82e3bf439621ac4d'; ...
    '6190eb8be0f39ad782e3bf439621ac4d'; ...
    'f4427ee23d4d51fabbf98dcec5f11066'; ...
    '6c0fb1806d66a9cc3002761e7a3ba3bd'; ...
    '59907bb6aaa7afeec283ca2c9df7372d'; ...
    '85b73c87c4c73263a7c64d6a7060b75b'; ...
    '9b52e73f96d181969cd431573238602d'; ...
    % '5c5f434f9ea8bf777bcb070cc655f13a'; ...
    '6dd44ada2f481c776dddf6f75fbd4a4c'; ...
    % 'a07b5b7014264e0330e06a011e63236a'; ...
    % 'e27d9fece990d39a0f23466c3c25e2e'; ...
    % '1c7ce8d5874553ccc3bd24f986301745' ...
    };

for i = 1:length(model_ids)
    filename = sprintf(fullfile(nrrd_dir, '%s/%s.nrrd'), model_ids{i}, model_ids{i});
    highres_filename = sprintf(fullfile(highres_nrrd_dir, '%s/%s.nrrd'), model_ids{i}, model_ids{i});
    
    %% Read NRRD file
    [X, meta] = nrrdread(filename);
    X = permute(X, [3, 4, 2, 1]);
    
    Xrgb = X(:,:,:,1:3);
    Xa = X(:, :, :, 4);
    figure;
    vol3d('CData', Xrgb, 'Alpha', Xa);
    title('Without filtering')
    
    %% Filtered high resolution and smartly sampled
    
    [X, meta] = nrrdread(highres_filename);
    X = permute(X, [3, 4, 2, 1]);
    
    Xrgb = X(:,:,:,1:3);
    Xa = X(:, :, :, 4);
    
    figure;
    vol3d('CData', Xrgb, 'Alpha', Xa);
    title('Highres (unmodified)')
    
    X_filtered = lp_filter(X);
    X_r_filtered = X_filtered(:, :, :, 1);
    X_g_filtered = X_filtered(:, :, :, 2);
    X_b_filtered = X_filtered(:, :, :, 3);
    
    Xrgb_filtered = cat(4, X_r_filtered, X_g_filtered, X_b_filtered);
    
    X_highres_filtered = cat(4, Xrgb_filtered, Xa);
    
    % figure;
    % vol3d('CData', X_highres_filtered(:, :, :, 1:3), 'Alpha', X_highres_filtered(:, :, :, 4));
    % title('Highres filtered but not sampled yet')
    
    X_subsampled = subsample(X_highres_filtered, output_res);
    Xrgb_filtered_subsampled = X_subsampled(:,:,:,1:3);
    Xa_sampled = X_subsampled(:, :, :, 4);
    
    figure;
    vol3d('CData', Xrgb_filtered_subsampled, 'Alpha', Xa_sampled);
    title('With filtering')
end