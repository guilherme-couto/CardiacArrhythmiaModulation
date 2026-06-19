//
// Created by Guilherme Couto on 25-12-2025. (Merry Christmas! São Miguel do Gostoso - RN, Brazil)
//

#include "../3dparty/stb_ds.h"
#include "../config_helpers/config_helpers.h"
#include "../libraries_common/common_data_structures.h"
#include "../logger/logger.h"
#include "../utils/stop_watch.h"
#include "../utils/utils.h"

#include "domain_helpers.h"
#include "mesh_info_data.h"

#include <time.h>
#include <unistd.h>
#include <float.h>
#include <math.h>


uint32_t set_custom_mesh_topology(struct grid *the_grid, const char *mesh_file, uint32_t num_volumes, double start_h) {

    struct stop_watch sw = {0};
    start_stop_watch(&sw);

    FILE *file = fopen(mesh_file, "r");

    if(!file) {
        log_error_and_exit("Error opening mesh described in %s!!\n", mesh_file);
    }

    struct custom_mesh_basic_data_hash_entry *custom_mesh_data_hash = NULL;
    hmdefault(custom_mesh_data_hash, -1);

    real_cpu maxx = 0.0, maxy = 0.0, maxz = 0.0;
    real_cpu minx = DBL_MAX, miny = DBL_MAX, minz = DBL_MAX;

    char *line = NULL;
    size_t len;

    log_info("Start - reading mesh topology (XYZ)\n");

    for(uint32_t i = 0; i < num_volumes; i++) {

        sds *data;
        int split_count;

        if (getline(&line, &len, file) == -1) {
             log_error_and_exit("Error reading line %d of mesh file %s\n", i, mesh_file);
        }

        char *tmp = line;
        data = sdssplit(tmp, ",", &split_count);

        if(split_count < 3) {
            log_error_and_exit("Not enough data to load the mesh geometry in line %d of file %s! [available=%d, required=3]\n", i + 1, mesh_file, split_count);
        }

        real_cpu cx = strtod(data[0], NULL);
        real_cpu cy = strtod(data[1], NULL);
        real_cpu cz = strtod(data[2], NULL);

        // Associate the spatial point with the file line index
        hmput(custom_mesh_data_hash, POINT3D(cx, cy, cz), i);

        if(cx > maxx) maxx = cx;
        if(cx < minx) minx = cx;
        if(cy > maxy) maxy = cy;
        if(cy < miny) miny = cy;
        if(cz > maxz) maxz = cz;
        if(cz < minz) minz = cz;

        sdsfreesplitres(data, split_count);
    }

    log_info("Finish - reading mesh topology\n");

    // Grid Construction
    double cube_side = start_h;
    double min_cube_side = fmax(maxx, fmax(maxy, maxz)) + start_h;

    while(cube_side < min_cube_side) {
        cube_side = cube_side * 2;
    }

    double tmp_size = cube_side / 2.0;
    uint16_t num_ref = 0;

    while(tmp_size > start_h) {
        tmp_size = tmp_size / 2;
        num_ref++;
    }

    initialize_and_construct_grid(the_grid, SAME_POINT3D(cube_side));

    uint32_t num_loaded = 0;

    struct point_3d min_bound = POINT3D(minx - start_h, miny - start_h, minz - start_h);
    struct point_3d max_bound = POINT3D(maxx + start_h, maxy + start_h, maxz + start_h);

    log_info("\nStart - refining the initial cube (refining %d times with a cube side of %lf)\n", num_ref, cube_side);
    refine_grid_with_bounds(the_grid, num_ref, min_bound, max_bound);
    log_info("Finish - refining the initial cube\n\n");

    log_info("Loading grid cells from hash map\n");

    FOR_EACH_CELL(the_grid) {
        real_cpu x = cell->center.x;
        real_cpu y = cell->center.y;
        real_cpu z = cell->center.z;

        if(x > maxx || y > maxy || z > maxz || x < minx || y < miny || z < minz) {
            cell->active = false;
        } else {
            struct point_3d p = POINT3D(x, y, z);
            int index = hmget(custom_mesh_data_hash, p);

            if(index != -1) {
                cell->active = true;
                cell->original_position_in_file = index;
                num_loaded++;
            } else {
                cell->active = false;
            }
        }
    }

    if(num_loaded > 0) {
        the_grid->mesh_side_length.x = maxx + start_h;
        the_grid->mesh_side_length.y = maxy + start_h;
        the_grid->mesh_side_length.z = maxz + start_h;

        log_info("Cleaning grid (derefining inactive)\n");
        for(uint16_t r = 0; r < num_ref; r++) {
            derefine_grid_inactive_cells(the_grid);
        }
    }

    free(line);
    hmfree(custom_mesh_data_hash);
    fclose(file);

    log_info("\nTime to load the mesh topology: %ld μs\n\n", stop_stop_watch(&sw));

    return num_loaded;
}

void load_fibrosis_fiber_and_border_zone_from_file(struct grid *the_grid, const char *mesh_file, uint32_t num_volumes) {
    
    log_info("Start - Loading extra fibrosis/fiber/border zone data from %s\n", mesh_file);

    struct stop_watch sw = {0};
    start_stop_watch(&sw);
    
    // Step 1: Create a mapping from "File Line Index" to "Cell Pointer"
    // This avoids O(N^2) searches or complex hash lookups during the file read.
    struct cell_node **cells_by_file_index = MALLOC_ARRAY_OF_TYPE(struct cell_node*, num_volumes);
    
    if(cells_by_file_index == NULL) {
        log_error_and_exit("Failed to allocate memory for cell mapping\n");
    }

    // Initialize with NULL to be safe
    OMP(parallel for)
    for(uint32_t i=0; i<num_volumes; i++) cells_by_file_index[i] = NULL;

    // Populate the map
    FOR_EACH_CELL(the_grid) {
        if(cell->active) {
            uint32_t idx = cell->original_position_in_file;
            if(idx < num_volumes) {
                cells_by_file_index[idx] = cell;
            }
        }
    }

    // Step 2: Read the file sequentially
    FILE *file = fopen(mesh_file, "r");
    if(!file) {
        log_error_and_exit("Error opening mesh file for data loading: %s\n", mesh_file);
    }

    char *line = NULL;
    size_t len = 0;
    
    // Expected columns: 
    // [0]center_x, [1]center_y, [2]center_z, [3]dx/2, [4]dy/2, [5]dz/2, 
    // [6]fibro_tag, [7]fx, [8]fy, [9]fz, [10]bz_layer, [11]bz_mult

    for(uint32_t i = 0; i < num_volumes; i++) {
        
        if(getline(&line, &len, file) == -1) break;

        // If this line corresponds to an active cell in our grid
        struct cell_node *cell = cells_by_file_index[i];
        
        if(cell != NULL) {
            int split_count;
            sds *data = sdssplit(line, ",", &split_count);

            INITIALIZE_FIBROTIC_INFO(cell);
            
            // Check if we have enough columns (6 standard + 6 extra = 12)
            if(split_count >= 12) {
                
                // HERE IS THE CUSTOM PART
                // Explicitly parsing Mixed Types (Ints and Doubles)
                int fibro_tag = atoi(data[6]);
                real_cpu fiber_x = strtod(data[7], NULL);
                real_cpu fiber_y = strtod(data[8], NULL);
                real_cpu fiber_z = strtod(data[9], NULL);
                int bz_layer = atoi(data[10]);
                real_cpu bz_multiplier = strtod(data[11], NULL);

                // ASSIGN TO CELL
                FIBROTIC(cell) = (fibro_tag == 1);
                BORDER_ZONE(cell) = (bz_layer > 0);
                BZ_LAYER(cell) = bz_layer;
                BZ_MULTIPLIER(cell) = bz_multiplier;

                if(FIBROTIC(cell)) {
                    cell->active = false;
                }

                cell->sigma.fibers.f[0] = fiber_x;
                cell->sigma.fibers.f[1] = fiber_y;
                cell->sigma.fibers.f[2] = fiber_z;
                
            } else {
                log_warn("Line %d has insufficient columns for fibrosis data. Expected 12, got %d.\n", i, split_count);
            }

            sdsfreesplitres(data, split_count);
        }
    }

    free(line);
    free(cells_by_file_index);
    fclose(file);
    
    log_info("\nTime to load fibrosis/fiber/border zone data: %ld μs\n\n", stop_stop_watch(&sw));
}

SET_SPATIAL_DOMAIN(initialize_grid_with_custom_mesh_with_fibrosis_fiber_and_border_zone) {

    char *mesh_file = NULL;
    GET_PARAMETER_STRING_VALUE_OR_REPORT_ERROR(mesh_file, config, "mesh_file");

    real_cpu start_h = 0.0;
    GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(real_cpu, start_h, config, "original_discretization");

    real_cpu max_h = start_h;
    GET_PARAMETER_NUMERIC_VALUE_OR_USE_DEFAULT(real_cpu, max_h, config, "maximum_discretization");

    uint32_t num_volumes = 0;
    GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(uint32_t, num_volumes, config, "num_volumes");

    uint32_t num_extra_fields = 0;
    GET_PARAMETER_NUMERIC_VALUE_OR_USE_DEFAULT(uint32_t, num_extra_fields, config, "num_extra_fields");

    the_grid->start_discretization = SAME_POINT3D(start_h);
    the_grid->max_discretization = SAME_POINT3D(max_h);

    // 1. Build the Spatial Domain (Topology Only)
    int num_loaded = (int) set_custom_mesh_topology(the_grid, mesh_file, num_volumes, start_h);

    // 2. Load Extra Data (Specific Logic)
    if(num_extra_fields > 0) {
        load_fibrosis_fiber_and_border_zone_from_file(the_grid, mesh_file, num_volumes);
    }

    free(mesh_file);
    return num_loaded;
}

void load_homogenized_fibrosis_data_from_file(struct grid *the_grid, const char *mesh_file, uint32_t num_volumes) {
    
    log_info("Start - Loading homogenized tensor and border zone data from %s\n", mesh_file);

    struct stop_watch sw = {0};
    start_stop_watch(&sw);
    
    // Step 1: Create a mapping from "File Line Index" to "Cell Pointer"
    struct cell_node **cells_by_file_index = MALLOC_ARRAY_OF_TYPE(struct cell_node*, num_volumes);
    
    if(cells_by_file_index == NULL) {
        log_error_and_exit("Failed to allocate memory for cell mapping\n");
    }

    OMP(parallel for)
    for(uint32_t i = 0; i < num_volumes; i++) cells_by_file_index[i] = NULL;

    FOR_EACH_CELL(the_grid) {
        if(cell->active) {
            uint32_t idx = cell->original_position_in_file;
            if(idx < num_volumes) {
                cells_by_file_index[idx] = cell;
            }
        }
    }

    // Step 2: Read the file sequentially
    FILE *file = fopen(mesh_file, "r");
    if(!file) {
        log_error_and_exit("Error opening mesh file for data loading: %s\n", mesh_file);
    }

    char *line = NULL;
    size_t len = 0;
    
    // Expected columns for Homogenized Mesh (15 columns): 
    // [0]cx, [1]cy, [2]cz, [3]hx, [4]hy, [5]hz, 
    // [6]fibro_tag, [7]Dxx, [8]Dyy, [9]Dzz, [10]Dxy, [11]Dxz, [12]Dyz, [13]bz_layer, [14]bz_mult

    for(uint32_t i = 0; i < num_volumes; i++) {
        
        if(getline(&line, &len, file) == -1) break;

        struct cell_node *cell = cells_by_file_index[i];
        
        if(cell != NULL) {
            int split_count;
            sds *data = sdssplit(line, ",", &split_count);

            INITIALIZE_FIBROTIC_INFO(cell);
            
            // Check if we have the 15 columns expected from the homogenized GNT output
            if(split_count >= 15) {
                
                int fibro_tag          = atoi(data[6]);
                real_cpu dxx           = strtod(data[7], NULL);
                real_cpu dyy           = strtod(data[8], NULL);
                real_cpu dzz           = strtod(data[9], NULL);
                real_cpu dxy           = strtod(data[10], NULL);
                real_cpu dxz           = strtod(data[11], NULL);
                real_cpu dyz           = strtod(data[12], NULL);
                int bz_layer           = atoi(data[13]);
                real_cpu bz_multiplier = strtod(data[14], NULL);

                // Assign basic info
                FIBROTIC(cell) = (fibro_tag == 1);
                BORDER_ZONE(cell) = (bz_layer > 0);
                BZ_LAYER(cell) = bz_layer;
                BZ_MULTIPLIER(cell) = bz_multiplier;

                // If the block is 100% blocked (fibro_tag == 1), deactivate it.
                if(FIBROTIC(cell)) {
                    cell->active = false;
                }

                // Directly inject the Effective Tensor into the cell's sigma structure
                cell->sigma.x  = dxx;
                cell->sigma.y  = dyy;
                cell->sigma.z  = dzz;
                cell->sigma.xy = dxy;
                cell->sigma.xz = dxz;
                cell->sigma.yz = dyz;
                
            } else {
                log_warn("Line %d has insufficient columns for homogenized data. Expected 15, got %d.\n", i, split_count);
            }

            sdsfreesplitres(data, split_count);
        }
    }

    free(line);
    free(cells_by_file_index);
    fclose(file);
    
    log_info("\nTime to load homogenized data: %ld μs\n\n", stop_stop_watch(&sw));
}

SET_SPATIAL_DOMAIN(initialize_grid_with_homogenized_mesh) {

    char *mesh_file = NULL;
    GET_PARAMETER_STRING_VALUE_OR_REPORT_ERROR(mesh_file, config, "mesh_file");

    real_cpu start_h = 0.0;
    GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(real_cpu, start_h, config, "original_discretization");

    real_cpu max_h = start_h;
    GET_PARAMETER_NUMERIC_VALUE_OR_USE_DEFAULT(real_cpu, max_h, config, "maximum_discretization");

    uint32_t num_volumes = 0;
    GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(uint32_t, num_volumes, config, "num_volumes");

    the_grid->start_discretization = SAME_POINT3D(start_h);
    the_grid->max_discretization = SAME_POINT3D(max_h);

    // 1. Build the Spatial Domain (Topology) using your existing topology loader
    int num_loaded = (int) set_custom_mesh_topology(the_grid, mesh_file, num_volumes, start_h);

    // 2. Load the Homogenized Tensors and BZ Data
    load_homogenized_fibrosis_data_from_file(the_grid, mesh_file, num_volumes);

    free(mesh_file);
    return num_loaded;
}