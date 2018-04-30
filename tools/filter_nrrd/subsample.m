function X_sampled = subsample(X, output_res)

input_res = size(X, 1);
Xa = X(:, :, :, 4);

factor = input_res / output_res;

X_sampled = zeros(output_res, output_res, output_res, 4, 'uint8');

for x = 1:output_res
    for y = 1:output_res
        for z = 1:output_res
            alpha_subtensor = Xa(((x-1)*factor + 1):(x*factor), ...
                                 ((y-1)*factor + 1):(y*factor), ...
                                 ((z-1)*factor + 1):(z*factor));
            occupied = any(alpha_subtensor(:));
            if occupied == true
                subtensor = X(((x-1)*factor + 1):(x*factor), ...
                              ((y-1)*factor + 1):(y*factor), ...
                              ((z-1)*factor + 1):(z*factor), ...
                              :);
                colors = get_occupied_colors(subtensor);
                cur_color = mean(colors, 1);
            else
                cur_color = [0, 0, 0];
            end
            X_sampled(x, y, z, :) = [cur_color, 255*occupied];
        end
    end
end

end