"""
Core utility module for FibroStruct.
Contains morphological and topological functions.
"""
import gudhi
import numpy as np
from collections import deque
from scipy.ndimage import convolve
from skimage.measure import label as ski_label, regionprops, block_reduce

INVALID_NUMERIC = -9999.0  # Standard fallback value for failed or physically invalid calculations

def calculate_cluster_metrics(fibrosis_mask):
    """
    Calculates Normalized Mean Cluster Size and Mean Aspect Ratio.
    Connectivity: Fully connected (Moore neighborhood, 8-way in 2D, 26-way in 3D).
    Note: The largest cluster is ALWAYS excluded from the mean size calculation.
    """
    f_labels = ski_label(fibrosis_mask, connectivity=2)
    f_props = regionprops(f_labels)
    num_comp = len(f_props)
    
    mean_size = INVALID_NUMERIC
    mean_aspect_ratio = INVALID_NUMERIC
    
    if num_comp > 0:
        f_props_sorted = sorted(f_props, key=lambda r: r.area, reverse=True)
        
        # Always exclude the largest cluster
        finite_clusters = [p.area for p in f_props_sorted[1:]] if num_comp > 1 else []
        
        if len(finite_clusters) > 0:
            sum_s2 = np.sum([s**2 for s in finite_clusters])
            sum_s = np.sum(finite_clusters)
            if sum_s > 0:
                mean_size = sum_s2 / sum_s
                
        # Aspect Ratio (weighted by area) across all clusters
        aspect_ratios = []
        cluster_areas = []
        for p in f_props:
            if len(p.bbox) == 4: # 2D
                h = p.bbox[2] - p.bbox[0]; w = p.bbox[3] - p.bbox[1]
                aspect_ratios.append(max(h, w) / max(min(h, w), 1))
            else: # 3D
                d = p.bbox[3] - p.bbox[0]; h = p.bbox[4] - p.bbox[1]; w = p.bbox[5] - p.bbox[2]
                dims = sorted([d, h, w])
                aspect_ratios.append(dims[2] / max(dims[0], 1))
            cluster_areas.append(p.area)
            
        if aspect_ratios:
            mean_aspect_ratio = np.average(aspect_ratios, weights=cluster_areas)

    return mean_size, mean_aspect_ratio

def calculate_fractal_dimension(fibrosis_mask):
    """
    Calculates Fractal Dimension using the standard Box-Counting method.
    """
    Z = fibrosis_mask
    assert(len(Z.shape) in [2, 3])
    p = min(Z.shape)
    if p == 0: return INVALID_NUMERIC
    
    n = int(np.floor(np.log2(p)))
    sizes = 2**np.arange(n, 1, -1)
    counts = []
    
    for size in sizes:
        if len(Z.shape) == 2:
            reduced = block_reduce(Z, block_size=(size, size), func=np.max, cval=0)
        else:
            reduced = block_reduce(Z, block_size=(size, size, size), func=np.max, cval=0)
        counts.append(np.sum(reduced))
    
    if len(counts) < 2: return INVALID_NUMERIC
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def create_inverse_distance_weight_matrix(ndim, radius=2):
    """Generates an inverse Euclidean distance weight matrix (1/d) for Moran's I."""
    size = 2 * radius + 1
    shape = tuple([size] * ndim)
    grids = np.indices(shape) - radius
    dist_sq = np.sum(grids**2, axis=0)
    W = np.zeros(shape, dtype=float)
    mask = dist_sq > 0
    W[mask] = 1.0 / np.sqrt(dist_sq[mask])
    return W

def calculate_morans_i(fibrosis_mask):
    """
    Calculates Spatial Autocorrelation (Moran's I) using inverse distance weighting.
    Connectivity: 5x5 (2D) or 5x5x5 (3D) window based on Euclidean distance decay.
    """
    ndim = fibrosis_mask.ndim
    W_kernel = create_inverse_distance_weight_matrix(ndim, radius=2)
    
    N = fibrosis_mask.size
    X_mean = np.mean(fibrosis_mask)
    Z = fibrosis_mask.astype(float) - X_mean
    Z2_sum = np.sum(Z**2)
    W_sum = convolve(np.ones_like(fibrosis_mask, dtype=float), W_kernel, mode='constant', cval=0.0).sum()
    
    if Z2_sum > 0 and W_sum > 0:
        WZ = convolve(Z, W_kernel, mode='constant', cval=0.0)
        lisa_map = Z * WZ 
        morans_i = (N / W_sum) * (np.sum(lisa_map) / Z2_sum)
        return morans_i
        
    return INVALID_NUMERIC

def calculate_betti_numbers(fibrosis_mask, dim_mode):
    """
    Computes Persistent Homology Betti Numbers using Gudhi Cubical Complex.
    Filtration: Healthy tissue (0.0) -> Space; Fibrosis (1.0) -> Obstacles/Holes.
    """
    try:
        filt_array = np.where(fibrosis_mask == 0, 0.0, 1.0)
        cc = gudhi.CubicalComplex(top_dimensional_cells=filt_array)
        cc.compute_persistence()
        betti = cc.persistent_betti_numbers(0.0, 0.0)
        
        betti_0 = betti[0] if len(betti) > 0 else 0 
        betti_1 = betti[1] if len(betti) > 1 else 0 
        betti_2 = betti[2] if dim_mode == "3D" and len(betti) > 2 else INVALID_NUMERIC
        
        return betti_0, betti_1, betti_2
    except Exception as e:
        return INVALID_NUMERIC, INVALID_NUMERIC, INVALID_NUMERIC

def run_bfs_x(healthy_mask, reconstruct_paths=False):
    """
    Executes Breadth-First Search strictly from Left (X=0) to Right (X=max).
    Connectivity: Von Neumann (4-way in 2D, 6-way in 3D).
    Returns percolation boolean, min path length, average path length, and visualization masks.
    """
    ndim = healthy_mask.ndim
    queue = deque()
    
    if ndim == 2:
        rows, cols = healthy_mask.shape
        distances = np.full((rows, cols), -1.0)
        if reconstruct_paths: parent = np.full((rows, cols, 2), -1, dtype=int)
        end_idx = cols - 1
        for r in range(rows):
            if healthy_mask[r, 0]: 
                queue.append((r, 0))
                distances[r, 0] = 0.0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        depths, rows, cols = healthy_mask.shape
        distances = np.full((depths, rows, cols), -1.0)
        if reconstruct_paths: parent = np.full((depths, rows, cols, 3), -1, dtype=int)
        end_idx = cols - 1
        for z in range(depths):
            for y in range(rows):
                if healthy_mask[z, y, 0]: 
                    queue.append((z, y, 0))
                    distances[z, y, 0] = 0.0
        directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

    while queue:
        if ndim == 2:
            r, c = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if healthy_mask[nr, nc] and distances[nr, nc] == -1.0:
                        distances[nr, nc] = distances[r, c] + 1.0
                        if reconstruct_paths: parent[nr, nc] = [r, c]
                        queue.append((nr, nc))
        else:
            z, y, x = queue.popleft()
            for dz, dy, dx in directions:
                nz, ny, nx = z + dz, y + dy, x + dx
                if 0 <= nz < depths and 0 <= ny < rows and 0 <= nx < cols:
                    if healthy_mask[nz, ny, nx] and distances[nz, ny, nx] == -1.0:
                        distances[nz, ny, nx] = distances[z, y, x] + 1.0
                        if reconstruct_paths: parent[nz, ny, nx] = [z, y, x]
                        queue.append((nz, ny, nx))

    percolates = False
    min_len = INVALID_NUMERIC
    avg_len = INVALID_NUMERIC
    short_mask = None
    all_paths_grid = None

    if ndim == 2:
        target_coords = [(r, end_idx) for r in range(rows) if distances[r, end_idx] > -1.0]
    else:
        target_coords = [(z, y, end_idx) for z in range(depths) for y in range(rows) if distances[z, y, end_idx] > -1.0]

    if len(target_coords) > 0:
        percolates = True
        valid_dists = [distances[coord] for coord in target_coords]
        min_len = np.min(valid_dists)
        avg_len = np.mean(valid_dists)

        if reconstruct_paths:
            short_mask = np.zeros_like(healthy_mask, dtype=bool)
            all_paths_grid = np.full_like(healthy_mask, np.nan, dtype=float)
            
            min_idx = np.argmin(valid_dists)
            shortest_target = target_coords[min_idx]

            for target in target_coords:
                dist = distances[target]
                curr = target
                is_shortest = (target == shortest_target)
                
                while True:
                    if ndim == 2:
                        r, c = curr
                        if np.isnan(all_paths_grid[r, c]) or dist < all_paths_grid[r, c]:
                            all_paths_grid[r, c] = dist
                        if is_shortest: short_mask[r, c] = True
                        pr, pc = parent[r, c]
                        if pr == -1 and pc == -1: break
                        curr = (pr, pc)
                    else:
                        z, y, x = curr
                        if np.isnan(all_paths_grid[z, y, x]) or dist < all_paths_grid[z, y, x]:
                            all_paths_grid[z, y, x] = dist
                        if is_shortest: short_mask[z, y, x] = True
                        pz, py, px = parent[z, y, x]
                        if pz == -1 and py == -1 and px == -1: break
                        curr = (pz, py, px)

    return percolates, min_len, avg_len, short_mask, all_paths_grid
