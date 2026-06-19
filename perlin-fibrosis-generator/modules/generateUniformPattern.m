function final_result = generateUniformPattern(geometry, target_density, seed, bz_cfg)
% Applies a purely random (stochastic) uniform distribution of fibrosis.
% Does not use Perlin noise. Guarantees exact target density.

    rng(seed);

    % Initialize matrices
    target_dims = size(geometry.core_mask);
    presence = false(target_dims);
    
    % Dummy threshold map to maintain compatibility with other modes and visualization
    % 999 represents healthy tissue
    threshold_map = ones(target_dims) * 999; 

    % 1. Core Generation (Exact density targeting)
    num_target_pixels = round(target_density * geometry.num_core_points);
    idx_rand = randperm(geometry.num_core_points, num_target_pixels);

    core_pixels = false(geometry.num_core_points, 1);
    core_pixels(idx_rand) = true;
    presence(geometry.core_indices) = core_pixels;

    % Viz compatibility: equivalent probability map
    threshold_map(geometry.core_mask) = 1.0 - target_density;

    % % 2. Border Zone Generation
    % if geometry.total_layers > 0
    %     rate = bz_cfg.threshold_rate;
    %     for layer = 1:geometry.total_layers
    %         % Decay the probability (density) exactly like the threshold logic
    %         progression = (layer / geometry.total_layers) ^ rate;
    %         layer_density = target_density * (1.0 - progression);

    %         mask_layer = (geometry.bz_map == layer);
    %         num_layer_points = sum(mask_layer(:));

    %         if num_layer_points > 0
    %             num_layer_target = round(layer_density * num_layer_points);
    %             layer_idx = find(mask_layer);
    %             idx_rand_layer = randperm(num_layer_points, num_layer_target);

    %             layer_pixels = false(num_layer_points, 1);
    %             layer_pixels(idx_rand_layer) = true;

    %             presence(layer_idx) = layer_pixels;
    %         end
            
    %         % Viz compatibility
    %         threshold_map(mask_layer) = 1.0 - layer_density;
    %     end
    % end

    % 2. Border Zone Generation
    if geometry.total_layers > 0
        % Exponential decay constant (Higher = faster drop in fibrosis)
        % 3.0 means it drops to ~5% by the outermost layer
        decay_factor = 3.0; 
        
        for layer = 1:geometry.total_layers
            % Normalized distance from core (0 to 1)
            progression = layer / geometry.total_layers;
            
            % Exponential decay formula
            layer_density = target_density * exp(-decay_factor * progression);

            mask_layer = (geometry.bz_map == layer);
            num_layer_points = sum(mask_layer(:));

            if num_layer_points > 0
                num_layer_target = round(layer_density * num_layer_points);
                layer_idx = find(mask_layer);
                idx_rand_layer = randperm(num_layer_points, num_layer_target);

                layer_pixels = false(num_layer_points, 1);
                layer_pixels(idx_rand_layer) = true;

                presence(layer_idx) = layer_pixels;
            end
            
            % Viz compatibility
            threshold_map(mask_layer) = 1.0 - layer_density;
        end
    end

    % 3. Package Result
    final_result.presence = presence;
    final_result.bz_layers = geometry.bz_map;
    final_result.fc_density = sum(presence(geometry.core_indices)) / geometry.num_core_points;
    final_result.threshold_map = threshold_map;
end