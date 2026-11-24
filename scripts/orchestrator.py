"""
This script orchestrates the entire pipeline for generating fibrotic meshes, calculating structural metrics,
running electrophysiological simulations with MonoAlg3D, and compiling results into a master dataset.

It is designed to be flexible for both desktop and cluster environments, with robust logging and error handling.
"""
import os
import subprocess
import configparser
import logging
import argparse
import shutil
import sys
import time
import uuid
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Import metrics functions from local library
from fibro_metrics import (
    calculate_cluster_metrics, calculate_fractal_dimension, calculate_morans_i,
    calculate_betti_numbers, run_bfs_x, INVALID_NUMERIC
)

warnings.filterwarnings('ignore')


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OCTAVE_SCRIPT_DIR = PROJECT_ROOT / "perlin-fibrosis-generator"
MONOALG3D_DIR = PROJECT_ROOT / "MonoAlg3D_C"
MONOALG_EXE = MONOALG3D_DIR / "bin" / "MonoAlg3D"


# HELPER CLASSES AND FUNCTIONS
class CasePreservingConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr): return optionstr

def setup_logging(log_file=None):
    """
    Configures logging to console and optionally to a file:
    - Console (INFO+): Clean summary, no clutter.
    - File (DEBUG+): Detailed audit trail of every seed.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    if root_logger.hasHandlers(): root_logger.handlers.clear()

    # Console Handler (Summary)
    console_fmt = logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_fmt)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # File Handler (Detailed)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setFormatter(file_fmt)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)

def print_progress(iteration, total, prefix='', suffix='', length=40):
    is_interactive = sys.stdout.isatty()
    percent = 100 * (iteration / float(total))
    if is_interactive:
        filled = int(length * iteration // total)
        bar = '█' * filled + '-' * (length - filled)
        sys.stdout.write(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}')
        if iteration == total: sys.stdout.write('\n')
        sys.stdout.flush()
    else:
        if iteration == 1 or iteration == total or iteration % max(1, total // 10) == 0:
            logging.info(f"{prefix} Progress: {iteration}/{total} ({percent:.1f}%) {suffix}")


# I/O AND SUBPROCESSES
def generate_fibrosis_mesh(mesh_config, output_batch_dir):
    """Calls the framework to generate the .alg file."""
    f_type = mesh_config["fibrosis_type"]
    angle_deg = mesh_config["angle_deg"]
    density = mesh_config["density"]
    seed = mesh_config["seed"]
    dim_mode = mesh_config["dim_mode"]
    domain_vec = mesh_config["domain_dims"]
    shape = mesh_config["shape"]
    core_vec = mesh_config["core_dims"]
    
    domain_str = ','.join(map(str, domain_vec))
    core_str = ','.join(map(str, core_vec)) if core_vec else ''

    tmp_id = uuid.uuid4().hex[:8]
    output_dir = Path(output_batch_dir) / f_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename_no_ext = f"{f_type}_angle{angle_deg}_density{density:.2f}_seed{seed}_{tmp_id}"
    full_output_path = output_dir / filename_no_ext
    final_mesh_path = full_output_path.with_suffix(full_output_path.suffix + ".alg")

    if final_mesh_path.exists(): return str(final_mesh_path)
    logging.debug(f"Generating mesh via Octave: {final_mesh_path.name}")

    # Call Octave script with parameters
    octave_cmd = f"run_fibrosis_generator('{f_type}', {density}, {seed}, {angle_deg}, '{dim_mode}', [{domain_str}], '{shape}', [{core_str}], '{str(full_output_path)}', true, false);"
    cmd = ["octave-cli", "-W", "-q", "--no-gui", "--eval", octave_cmd]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(OCTAVE_SCRIPT_DIR))
    except subprocess.CalledProcessError as e:
        logging.error(f"Octave execution failed for seed {seed}!")
        logging.error(f"STDOUT: {e.stdout}")
        logging.error(f"STDERR: {e.stderr}")
        raise e

    return str(final_mesh_path)

def parse_alg_to_grid(mesh_path, dim_mode="2D"):
    """Parses the .alg file into a binary Numpy array (True = Conducting, False = Fibrosis)."""
    try:
        # Read the .alg file assuming the format: x, y, z, dx/2, dy/2, dz/2, fibro_tag, ...
        df = pd.read_csv(mesh_path, header=None)
        x_coords, y_coords = df.iloc[:, 0].values, df.iloc[:, 1].values
        tags = df.iloc[:, 6].values
        
        # Create unique sorted coordinate arrays and index mappings
        unique_x, unique_y = np.sort(np.unique(x_coords)), np.sort(np.unique(y_coords))
        x_idx, y_idx = np.searchsorted(unique_x, x_coords), np.searchsorted(unique_y, y_coords)
        
        if dim_mode == "3D":
            z_coords = df.iloc[:, 2].values
            unique_z = np.sort(np.unique(z_coords))
            if len(unique_x) < 2 or len(unique_y) < 2 or len(unique_z) < 2: return None
            z_idx = np.searchsorted(unique_z, z_coords)
            grid = np.ones((len(unique_z), len(unique_y), len(unique_x))) * -1 
            grid[z_idx, y_idx, x_idx] = tags
        else:
            if len(unique_x) < 2 or len(unique_y) < 2: return None
            grid = np.ones((len(unique_y), len(unique_x))) * -1 
            grid[y_idx, x_idx] = tags
            
        # Assuming tag 0 = Conducting, 1 = Fibrosis. Return True for Conducting, False for Fibrosis
        return (grid == 0)
    except Exception:
        return None
    
def configure_and_run_simulation(mesh_path, sim_output_dir, sim_params, batch_dir, base_config_path):
    """Configures the .ini file and runs the MonoAlg3D simulation."""
    sim_output_dir.mkdir(parents=True, exist_ok=True)

    ftype = sim_params['fibrosis_type']
    angle_deg = sim_params['fiber_angle_deg']
    density = sim_params['density']
    seed = sim_params['seed']
    atpi_value = sim_params['atpi']

    # Parse ALG header for discretization
    with open(mesh_path, 'r') as f:
        first_line = f.readline()
        lines_count = sum(1 for _ in f) + 1 # +1 for the line we just read
        parts = first_line.split(',')
        half_dx = float(parts[3].strip()) # Column 3 (0-index 3) is dx/2 in output format
        num_extra_fields = len(parts) - 6

    # Configure .ini
    config = CasePreservingConfigParser()
    config.read(base_config_path)
    simulation_name = f"{ftype} angle {angle_deg} den {density} seed {seed} atpi {atpi_value}"
    config['domain']['mesh_file'] = str(mesh_path)
    config['domain']['num_volumes'] = str(lines_count)
    config['domain']['original_discretization'] = str(half_dx * 2)
    config['domain']['num_extra_fields'] = str(num_extra_fields)
    config['domain']['name'] = simulation_name
    config['save_result']['output_dir'] = str(sim_output_dir)
    config['extra_data']['atpi'] = str(atpi_value)

    ini_path = batch_dir / "configs" / f"{ftype}.ini"
    with open(ini_path, 'w') as configfile: config.write(configfile)

    # Execute the simulation with MonoAlg3D
    command = [str(MONOALG_EXE), "-c", str(ini_path)]

    logging.debug(f"Launching MonoAlg3D: {' '.join(command)}")

    try:
        subprocess.run(command, check=True, capture_output=True, text=True, cwd=MONOALG3D_DIR)
    except subprocess.CalledProcessError as e:
        logging.error(f"MonoAlg3D failed for {simulation_name}.")
        logging.error(f"Stderr: {e.stderr}")
        raise e
    
def parse_simulation_log(sim_output_dir):
    """Parses outputlog.txt for final time."""
    simulation_log_file = sim_output_dir / "outputlog.txt"
    if not simulation_log_file.is_file():
        logging.error(f"Output log not found: {simulation_log_file}")
        return -1.0

    last_t = 0.0
    with open(simulation_log_file, 'r') as f:
        for line in f:
            if line.startswith('t = '):
                try:
                    t_value = float(line.split(',')[0].split('=')[1].strip())
                    if t_value > last_t: last_t = t_value
                except (IndexError, ValueError):
                    logging.error(f"Parse error in {simulation_log_file.name}: {line.strip()}")
                    raise

    return last_t

def cleanup_simulation_files(mesh_path, sim_output_dir):
    """Deletes the .alg mesh and clears the simulation output directory, keeping only essential logs to save disk space."""
    # Delete the generated mesh
    mesh_file = Path(mesh_path)
    if mesh_file.exists():
        try:
            mesh_file.unlink()
        except Exception as e:
            logging.warning(f"Failed to delete mesh {mesh_file.name}: {e}")

    # Clean the simulation output directory
    files_to_keep = {'original_configuration.ini', 'outputlog.txt', 'ecg.txt'}
    if sim_output_dir.exists():
        for item in sim_output_dir.iterdir():
            if item.name not in files_to_keep:
                try: shutil.rmtree(item) if item.is_dir() else item.unlink()
                except Exception as e:
                    logging.warning(f"Failed to delete output item {item.name}: {e}")


# WORKER
def process_single_seed(seed, angle_deg, density, mesh_config, batch_dir, base_config_path, shape, atpi_val, keep_meshes=False):
    """
    1. Generates Mesh -> 2. Calculates Structure -> 3. Runs EP Solver -> 4. Deletes Mesh.
    """
    mesh_path = None
    sim_out_dir = None
    try:
        local_config = mesh_config.copy()
        local_config["angle_deg"] = angle_deg
        local_config["density"] = density
        local_config["seed"] = seed
        ftype = local_config["fibrosis_type"]
        dim_mode = local_config["dim_mode"]
        dx = local_config["domain_dims"][0]
        
        # 1. Mesh Generation (or Load)
        alg_meshes_dir = batch_dir / "alg_meshes"
        mesh_path = generate_fibrosis_mesh(local_config, alg_meshes_dir)

        # Read the generated mesh to create masks for structural analysis
        conducting_mask = parse_alg_to_grid(mesh_path, dim_mode=dim_mode)
        if conducting_mask is None: return {"seed": seed, "error": "Failed to parse mesh or insufficient dimensions."}

        # Dimension Validation
        expected_x = int(round(local_config["domain_dims"][1] / dx))
        expected_y = int(round(local_config["domain_dims"][2] / dx))
        if dim_mode == "2D":
            rows, cols = conducting_mask.shape
            depths = INVALID_NUMERIC
            if cols != expected_x or rows != expected_y:
                return {"seed": seed, "error": f"Corrupted mesh grid parsed. Expected {expected_x}x{expected_y}, got {cols}x{rows}."}
        else:
            depths, rows, cols = conducting_mask.shape
            expected_z = int(round(local_config["domain_dims"][3] / dx))
            if cols != expected_x or rows != expected_y or depths != expected_z:
                return {"seed": seed, "error": f"Corrupted mesh grid parsed. Expected {expected_x}x{expected_y}x{expected_z}, got {cols}x{rows}x{depths}."}

        fibrosis_mask = (~conducting_mask).astype(int)
        total_cells = fibrosis_mask.size
        actual_density = np.sum(fibrosis_mask) / total_cells if total_cells > 0 else 0.0
        
        # Base structure
        res = {
            "fibrosis_type": ftype, "fiber_angle_deg": angle_deg, "density": density, "actual_density": f"{actual_density:.5f}", "seed": seed,
            "shape": shape, "dx": dx, "grid_x": cols, "grid_y": rows, "grid_z": depths, "atpi": atpi_val,
            "error": None
        }

        # 2. Structural Metrics Analysis
        mean_cluster_size, mean_aspect_ratio = calculate_cluster_metrics(fibrosis_mask)
        norm_mean_cluster_size = (mean_cluster_size / (cols * rows * (depths if dim_mode == "3D" else 1))) if mean_cluster_size != INVALID_NUMERIC else INVALID_NUMERIC
        res["norm_mean_cluster_size"] = norm_mean_cluster_size
        res["mean_aspect_ratio"] = mean_aspect_ratio
        res["fractal_dimension"] = calculate_fractal_dimension(fibrosis_mask)
        res["morans_i"] = calculate_morans_i(fibrosis_mask)
        res["betti_0"], res["betti_1"], res["betti_2"] = calculate_betti_numbers(fibrosis_mask, dim_mode)

        percolates_x, min_path_x, avg_path_x, short_mask, all_paths_grid = run_bfs_x(conducting_mask, reconstruct_paths=False)
        res["percolates_x"] = 1 if percolates_x else 0
        res["min_path_x"] = min_path_x if percolates_x else INVALID_NUMERIC
        res["topological_conduction_eff"] = (cols / avg_path_x) if percolates_x else INVALID_NUMERIC

        # 3. Electrophysiology Simulations (MonoAlg3D)
        sim_params = res.copy()
        sim_out_dir = batch_dir / "outputs" / f"atpi{atpi_val}" / f"{ftype}_angle{angle_deg}" / f"den{density}_seed{seed}"
        configure_and_run_simulation(mesh_path, sim_out_dir, sim_params, batch_dir, base_config_path)
        
        # Results Parsing
        final_time = parse_simulation_log(sim_out_dir)
        if final_time >= 0:
            res['final_time_ms'] = final_time
            if final_time >= 1000.0:
                res['sustained_reentry'] = 1
            else:
                res['sustained_reentry'] = 0
        else:
            return {"seed": seed, "error": "MonoAlg3D failed or returned invalid time."}

        return res
        
    except Exception as e:
        return {"seed": seed, "error": str(e)}
        
    finally:
        # 4. Cleanup to save space (delete mesh and simulation outputs except logs)
        if not keep_meshes:
            cleanup_simulation_files(mesh_path, sim_out_dir)
     
    
# BATCH EXECUTION  
def run_simulation_batch(args, seeds, densities, mesh_config, batch_dir, base_config_path, results_csv_path):
    # Ensure CSV exists
    if not results_csv_path.exists():
        results_csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_cols = ['fibrosis_type', 'fiber_angle_deg', 'density', 'actual_density', 'seed',
                    'shape', 'dx', 'grid_x', 'grid_y', 'grid_z', 'atpi',
                    'norm_mean_cluster_size', 'mean_aspect_ratio', 'fractal_dimension', 'morans_i', 'betti_0', 'betti_1', 'betti_2', 
                    'percolates_x', 'min_path_x', 'topological_conduction_eff', 'final_time_ms', 'sustained_reentry']
        pd.DataFrame(columns=csv_cols).to_csv(results_csv_path, index=False)
        logging.debug(f"Created new results file at: {results_csv_path}")
        completed_df = pd.DataFrame(columns=csv_cols)
    else:
        completed_df = pd.read_csv(results_csv_path)
        logging.debug(f"Loaded history: {len(completed_df)} simulations found.")

    total_tasks = len(args.ftypes) * len(args.angles) * len(densities) * len(args.atpi_vals)
    task_idx = 0
    
    # Sequential loops
    for ftype in args.ftypes:
        ftype = ftype.capitalize()
        mesh_config["fibrosis_type"] = ftype
        
        logging.info("-" * 60)
        logging.info(f"--> PROCESSING PHENOTYPE: {ftype}")
        logging.info("-" * 60)

        for atpi_val in args.atpi_vals:
            for angle in args.angles:
                for den in densities:
                    task_idx += 1
                    logging.info(f"[{task_idx}/{total_tasks}] ATPI={atpi_val} | Angle={angle}° | Density={den:.2f}")
                    
                    processed = 0
                    skipped = 0
                    errors = 0
                    start_time = time.time()
                    
                    for i, seed in enumerate(seeds):
                        print_progress(i + 1, len(seeds), prefix="Processing seeds:", suffix=f"(Run: {processed}, Skip: {skipped}, Err: {errors})", length=30)
                        
                        # Check if already processed
                        if not completed_df.empty:
                            q = completed_df[
                                (completed_df['fibrosis_type'] == ftype) & (completed_df['fiber_angle_deg'] == angle) &
                                (np.isclose(completed_df['density'], den, atol=1e-4)) & (completed_df['seed'] == seed) &
                                (completed_df['shape'] == args.shape) & (completed_df['atpi'] == atpi_val)
                            ]
                            if not q.empty:
                                skipped += 1
                                logging.debug(f"SEED {seed}: Skipped (Simulation already done)")
                                continue
                        
                        # Process sequentially
                        res = process_single_seed(seed, angle, den, mesh_config, batch_dir, base_config_path, args.shape, atpi_val, args.keep_meshes)
                        
                        if "error" in res and res["error"]:
                            errors += 1
                            logging.warning(f"SEED {seed} Failed: {res['error']}")
                        else:
                            processed += 1
                            logging.debug(f"SEED {seed}: Success (Sim finished: {res['final_time_ms']} ms -> {'with sustained reentry' if res.get('sustained_reentry', False) else 'no sustained reentry'})")
                            df_res = pd.DataFrame([res], columns=completed_df.columns)
                            df_res.to_csv(results_csv_path, mode='a', header=False, index=False)
                            completed_df = pd.concat([completed_df, df_res], ignore_index=True)

                    logging.info(f"   > Done in {time.time() - start_time:.1f}s. New: {processed}, Skip: {skipped}, Err: {errors}.")


# ENTRY POINT
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cardiac Arrhythmia Modulation Orchestrator (Structural Metrics + Electrophysiological Simulations).")
    parser.add_argument("--ftypes", type=str, nargs='+', required=True, choices=['compact', 'diffuse', 'interstitial', 'patchy', 'uniform', 'all'], help="Fibrosis types to simulate (choose from: compact, diffuse, interstitial, patchy, uniform, or 'all').")
    parser.add_argument("--base_config", type=str, required=True, help="Path to the base configuration .ini file for MonoAlg3D simulations.")
    parser.add_argument("--batch_dir", type=str, default=f"simulation_batches_{datetime.now().strftime('%Y%m%d_%H%M%S')}", help="Base directory for storing configurations, outputs, and results.")
    parser.add_argument("--angles", type=int, nargs='+', default=[0, 30, 60, 90], help="List of fiber angles in degrees to simulate.")
    parser.add_argument("--min_den", type=float, default=0.1, help="Minimum fibrosis density (between 0.0 and 1.0).")
    parser.add_argument("--max_den", type=float, default=0.9, help="Maximum fibrosis density (between 0.0 and 1.0).")
    parser.add_argument("--dens_step", type=float, default=0.05, help="Step size for fibrosis density values.")
    parser.add_argument("--densities", type=float, nargs='+', default=[], help="Explicit list of densities to simulate.")
    parser.add_argument("--num_seeds", type=int, default=100, help="Number of random seeds to use for simulations.")
    parser.add_argument("--atpi_vals", type=float, nargs='+', default=[2.0], help="List of [ATP]i values to simulate.")
    parser.add_argument("--dim_mode", type=str, default="2D", choices=["2D", "3D"], help="Dimension mode for simulations (2D or 3D).")
    parser.add_argument("--shape", type=str, default="full", help="Geometric shape for fibrosis distribution (e.g., 'ellipse', 'box', 'full').")
    parser.add_argument("--domain_dims", type=float, nargs='+', default=[0.01, 4.0, 4.0, 0.0], help="Domain dimensions as a vector: [dx, Lx, Ly, (Lz for 3D)].")
    parser.add_argument("--core_dims", type=float, nargs='+', default=[], help="Core dimensions for certain shapes as a vector: [width, height, depth]. Required if shape is not 'full'.")
    parser.add_argument("--keep_meshes", action="store_true", help="Keep generated mesh files (.alg) after simulations (default: False).")
    
    args = parser.parse_args()
    if 'all' in [t.lower() for t in args.ftypes]: args.ftypes = ['compact', 'diffuse', 'interstitial', 'patchy', 'uniform']
    args.ftypes = list(dict.fromkeys(args.ftypes))

    if not MONOALG_EXE.exists():
        print(f"CRITICAL ERROR: MonoAlg3D executable not found at {MONOALG_EXE}")
        print("Please compile MonoAlg3D_C before running the orchestrator.")
        sys.exit(1)

    scenario_batch_dir = PROJECT_ROOT / args.batch_dir / args.dim_mode / args.shape
    scenario_batch_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=scenario_batch_dir / "logs" / f"run_log_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    start_time = datetime.now()
    logging.info("="*60)
    logging.info(f"EXECUTION STARTED AT: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Host System: {os.name.upper()}")
    logging.info("="*60)
    
    # Verify base configuration file exists and copy to batch directory
    base_config_path = Path(args.base_config)
    if not base_config_path.is_file():
        logging.error(f"Base configuration file not found: {base_config_path}")
        sys.exit(1)
    dest_base = scenario_batch_dir / "configs" / "base_simulation_config.ini"
    dest_base.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(args.base_config), dest_base)

    # Range of densities
    density_step = args.dens_step
    simulation_densities = np.arange(args.min_den, args.max_den + density_step, density_step, dtype=float).tolist()
    if args.densities and len(args.densities) > 0:
        simulation_densities = args.densities
    simulation_densities = [round(d, 2) for d in simulation_densities if 0.0 < d < 1.0]
    simulation_densities = sorted(list(set(simulation_densities)))

    # Seeds
    master_seed = 20260101 
    rng = np.random.default_rng(master_seed)
    seeds = rng.integers(low=1, high=99999, size=args.num_seeds).tolist()
    # seeds = rng.choice(99999, size=args.num_seeds, replace=False).tolist() # Unique seeds only

    # Build Meshes Configuration
    mesh_config = {
        "dim_mode": args.dim_mode, "shape": args.shape, "domain_dims": args.domain_dims, 
        "core_dims": args.core_dims if args.shape.lower() != "full" else []
    }

    # Configuration Summary Log
    logging.info("--- CONFIGURATION SUMMARY")
    logging.info(f"Base Directory   : {scenario_batch_dir.resolve()}")
    logging.info(f"Config File      : {dest_base.name}")
    logging.info(f"Densities        : {simulation_densities}")
    logging.info(f"Fibrosis Type    : {args.ftypes} ({args.dim_mode} - {args.shape})")
    logging.info(f"Random Seeds     : {args.num_seeds} seeds generated (Master: {master_seed})")
    logging.debug(f"Full Seed List   : {seeds}")
    logging.debug(f"Meshes Config    : {mesh_config}")

    logging.info("-" * 30)

    try:
        csv_dataset_path = scenario_batch_dir / "data" / "master_dataset.csv"
        run_simulation_batch(args, seeds, simulation_densities, mesh_config, scenario_batch_dir, dest_base, csv_dataset_path)
    except KeyboardInterrupt:
        logging.warning("\nBatch execution interrupted by user (Ctrl+C).")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"Fatal execution error: {e}", exc_info=True)
        sys.exit(1)
        
    end_time = datetime.now()
    duration = end_time - start_time
    logging.info(f"EXECUTION FINISHED AT: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"TOTAL DURATION: {duration}")
