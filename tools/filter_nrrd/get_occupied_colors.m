function colors = get_occupied_colors(X)
Xrgb = X(:, :, :, 1:3); 
Xa = X(:, :, :, 4);
Xrgb_list = reshape(Xrgb, [size(Xrgb, 1)*size(Xrgb, 2)*size(Xrgb, 3), 3]);
Xa_list = Xa(:);
occupied_indices = find(Xa_list);
colors = Xrgb_list(occupied_indices, :);
end