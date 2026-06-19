//
// Created by Guilherme Couto on 25-12-2025. (Merry Christmas! São Miguel do Gostoso - RN, Brazil)
//

#include "../config/extra_data_config.h"
#include "../config_helpers/config_helpers.h"
#include "../libraries_common/common_data_structures.h"
#include "../domains_library/mesh_info_data.h"

#include "helper_functions.h"

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>


SET_EXTRA_DATA (set_extra_data_for_hypoxia_with_border_zone_multiplier_from_mesh) {
    
    uint32_t num_active_cells = the_grid->num_active_cells;
    struct cell_node ** ac = the_grid->active_cells;

    struct extra_data_for_fibrosis *extra_data = NULL;
    extra_data = set_common_schemia_data(config, num_active_cells);

    // Set border zone multiplier so hypoxia effect diminishes from the core to the border
    OMP(parallel for)
    for(int i = 0; i < num_active_cells; i++) {

        // BZ_MULTIPLIER goes from 0 (fibrotic core) to 1 (healthy tissue). Already set during mesh loading
        // extra_data->fibrosis[i] goes from 0 (fibrotic core - severe hypoxia) to 1 (healthy tissue - no hypoxia)
        extra_data->fibrosis[i] = BZ_MULTIPLIER(ac[i]);
        
    }

    SET_EXTRA_DATA_SIZE(sizeof(struct extra_data_for_fibrosis));

    return (void*)extra_data;
}
