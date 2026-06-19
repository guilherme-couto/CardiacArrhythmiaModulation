function result = applyFibrosisThreshold(noise_field, geometry, target_density, bz_cfg)
% Applies thresholding logic using the PRE-COMPUTED geometry to match specific densities.

    % Search parameters for Bisection Method
    max_iters = 40;
    tol = 1e-3;

    % 1. Determine Core Threshold
    % Only look at pixels inside the core
    core_values = noise_field(geometry.core_indices);
    [core_threshold, actual_core_density] = findThresholdForDensity(core_values, target_density, tol, max_iters);

    % 2. Build Full Threshold Map
    % Initialize with a value > 1 (impossible to cross for normalized noise)
    threshold_map = ones(size(noise_field)) * 999; 
    
    % Set Core
    threshold_map(geometry.core_mask) = core_threshold;
    
    % Set Border Zone Layers
    if geometry.total_layers > 0
        % You can also replace 3.0 with bz_cfg.decay_rate in the future if you want it configurable
        decay_factor = 3.0; % 3.0 means it drops to ~5% of the core density by the outermost layer
        L = geometry.total_layers;
        
        for layer = 1:L
            % Get mask and actual noise values for this specific layer
            mask_layer = (geometry.bz_map == layer);
            layer_values = noise_field(mask_layer);
            
            % Edge case: skip if layer is somehow empty
            if isempty(layer_values)
                continue;
            end
            
            % Calculate target density for this layer using the exponential decay rule
            % layer_density = core_density * exp(-3.0 * (l/L))
            target_layer_density = target_density * exp(-decay_factor * (layer / L));
            
            % Determine the exact threshold needed for this layer's noise distribution
            layer_thresh = findThresholdForDensity(layer_values, target_layer_density, tol, max_iters);
            
            % Assign to the threshold map
            threshold_map(mask_layer) = layer_thresh;
        end
    end
    
    % 3. Apply Threshold
    presence = (noise_field >= threshold_map);

    % 4. Reshape presence and threshold_map to dimensional matrices
    target_dims = size(geometry.core_mask);
    presence = reshape(presence, target_dims);
    threshold_map = reshape(threshold_map, target_dims);
    
    % 5. Package Result
    result.presence = presence;
    result.bz_layers = geometry.bz_map;
    result.fc_density = actual_core_density;
    result.threshold_map = threshold_map;
end

% =========================================================================
% Helper Functions
% =========================================================================

function [thresh, actual_dens] = findThresholdForDensity(values, target_dens, tol, max_iters)
% Finds the optimal threshold for a given array of noise values to reach a 
% specific target density using the Bisection Method.

    % Handle edge cases to prevent unnecessary loops
    if target_dens <= 0
        thresh = max(values) + 0.1; % Threshold above max = 0 density
        actual_dens = 0;
        return;
    end
    if target_dens >= 1
        thresh = min(values) - 0.1; % Threshold below min = 100% density
        actual_dens = 1;
        return;
    end

    low = min(values);
    high = max(values);
    
    for i = 1:max_iters
        thresh = (low + high) / 2;
        actual_dens = sum(values >= thresh) / length(values);
        
        if abs(actual_dens - target_dens) < tol
            break;
        end
        
        if actual_dens < target_dens
            high = thresh; % Need lower threshold to get more density
        else
            low = thresh;  % Need higher threshold to decrease density
        end
    end
end