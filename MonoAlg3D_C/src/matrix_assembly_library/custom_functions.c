#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../3dparty/sds/sds.h"
#include "../3dparty/stb_ds.h"
#include "../alg/grid/grid.h"
#include "../config/assembly_matrix_config.h"
#include "../libraries_common/common_data_structures.h"
#include "../utils/file_utils.h"
#include "../utils/utils.h"
#include "../domains_library/mesh_info_data.h"

#include "assembly_common.c"

#ifdef COMPILE_CUDA
#include "../gpu_utils/gpu_utils.h"
#endif



ASSEMBLY_MATRIX(homogenized_sigma_assembly_matrix) {

    if(the_grid->adaptive) {
        log_error_and_exit("homogenized_sigma_assembly_matrix does not support mesh adaptivity yet! Aborting!\n");
    }

    uint32_t num_active_cells = the_grid->num_active_cells;
    struct cell_node **ac = the_grid->active_cells;

    initialize_diagonal_elements(the_solver, the_grid);

    log_info("Assembling matrix using pre-computed effective homogenized tensors.\n");

    // The cell->sigma values (x, y, z, xy, xz, yz) are already populated by the domain loader
    // We only need to iterate over the active cells and calculate the face fluxes using the existing helper
    OMP(parallel for)
    for(uint32_t i = 0; i < num_active_cells; i++) {
        fill_discretization_matrix_elements_aniso(ac[i]);
    }
}