function [output_data] = generateFibrosisWorkflow(domain_cfg, geom_cfg, bz_cfg, noise_params, run_cfg)
% Orchestrates the GNT (Geometry-Noise-Threshold) pipeline.

    %% 1. BUILD MESH
    % Assuming you have a buildMesh function that returns struct
    mesh = buildMesh(domain_cfg);

    %% 2. COMPUTE GEOMETRY (The "G" in GNT)
    % Runs ONCE. Heavily optimized with bwdist.
    geometry = computeFibrosisGeometry(mesh, geom_cfg, bz_cfg);

    if geometry.num_core_points == 0
        error('Geometry generation failed: Core is empty.');
    end

    %% 3. GENERATE PATTERN (The "N" and "T" in GNT)
    if strcmpi(run_cfg.mode, 'single-pass')
        % Single Pass
        combined_noise = computeNoiseField(mesh, noise_params);
        final_result = applyFibrosisThreshold(combined_noise, geometry, run_cfg.target_density, bz_cfg);

        export_paper_data = false;
        if export_paper_data && ~mesh.is3D
            paper_data_file = sprintf('paper_viz_data_%s.mat', run_cfg.output_name);
            
            core_mask = geometry.core_mask;
            bz_map = geometry.bz_map;
            target_dims = size(geometry.core_mask);
            noise_field = reshape(combined_noise, target_dims);
            threshold_map = final_result.threshold_map;
            presence = final_result.presence;
            
            save("-v7", paper_data_file, "core_mask", "bz_map", "noise_field", "threshold_map", "presence");
            
            fprintf('Exported raw data for figure: %s\n', paper_data_file);
        end

    elseif strcmpi(run_cfg.mode, 'composition')
        % Iterative Loop (Uses the PRE-COMPUTED geometry)
        final_result = runCompositionLoop(mesh, geometry, noise_params, run_cfg.target_density, bz_cfg);

        export_paper_data = false;
        if export_paper_data && ~mesh.is3D
            paper_data_file = sprintf('paper_viz_data_%s.mat', run_cfg.output_name);
            
            % Generate ONE representative noise field to illustrate the texture
            noise_field = computeNoiseField(mesh, noise_params, noise_params.seed + 1);
            target_dims = size(geometry.core_mask);
            rep_noise_field = reshape(noise_field, target_dims);
            
            % Extract variables for saving
            core_mask = geometry.core_mask;
            bz_map = geometry.bz_map;
            presence = final_result.presence;
            
            save("-v7", paper_data_file, "core_mask", "bz_map", "rep_noise_field", "presence");
            fprintf('Exported raw data for composition figure: %s\n', paper_data_file);
        end
    
    elseif strcmpi(run_cfg.mode, 'uniform')
        % Purely random distribution
        final_result = generateUniformPattern(geometry, run_cfg.target_density, run_cfg.seed, bz_cfg);
        
        export_paper_data = false;
        if export_paper_data && ~mesh.is3D
            paper_data_file = sprintf('paper_viz_data_%s.mat', run_cfg.output_name);
            
            % Generate a white noise field purely for visualization purposes
            rng(run_cfg.seed);
            noise_field = rand(size(geometry.core_mask));
            
            core_mask = geometry.core_mask;
            bz_map = geometry.bz_map;
            threshold_map = final_result.threshold_map;
            presence = final_result.presence;
            
            save("-v7", paper_data_file, "core_mask", "bz_map", "noise_field", "threshold_map", "presence");
        end
        
    else
        error('Unknown mode: %s', run_cfg.mode);
    end

    %% 4. SAVE RESULTS
    % No flips, no permutes. buildMesh geometry (Row 1 = Bottom).
    % The data is passed exactly as generated in geometry: (Ny, Nx, [Nz])
    % Structuring output for saving
    output_data.presence = final_result.presence;
    output_data.bz_layers = final_result.bz_layers;
    output_data.mesh = mesh;
    output_data.params = noise_params;
    output_data.is3D = mesh.is3D;
    
    if run_cfg.save_mesh
        writeALGFromPatternData(output_data, run_cfg.output_name);
    end

    if run_cfg.save_fig
        savePatternVisualisation(output_data, final_result, run_cfg);
    end
end