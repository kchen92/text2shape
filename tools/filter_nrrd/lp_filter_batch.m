function lp_filter_batch(nrrd_dir, out_dir, output_res)

% nrrd_dir: Directory of high-resolution NRRDs
% out_dir: Directory for outputting the high-resolution NRRDs
% output_res: Scalar representing output resolution ([output_res] * 3)

addpath('../matlab')

model_ids = dir(nrrd_dir);
model_ids = model_ids(3:end, :);  % Get rid of '.' and '..' directories

mkdir(out_dir);

parfor i = 1:size(model_ids, 1)  % Change to parfor
    model_id = model_ids(i).name;
    cur_nrrd_filename = strcat(model_id, '.nrrd');
    relative_nrrd_path = fullfile(model_id, cur_nrrd_filename);
    cur_filepath = fullfile(nrrd_dir, relative_nrrd_path);
    
    try
        [X_subsampled, meta] = filter_and_sample(cur_filepath, output_res);
    catch
        disp(cur_filepath)
        continue
    end
    
    mkdir(fullfile(out_dir, model_id))
    
    out_filename = fullfile(out_dir, relative_nrrd_path);
    customNrrdWriter(out_filename, X_subsampled, 'gzip', meta);
end

end