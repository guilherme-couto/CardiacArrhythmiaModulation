"""
Comprehensive Unified Analysis Module for FibroStruct (2D and 3D).

Reads consolidated CSVs containing topological, spatial, and functional data.
Dynamically adapts to 2D or 3D datasets, handling explicit invalid data flags.
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from scipy.optimize import curve_fit, root_scalar

warnings.filterwarnings('ignore')

# GLOBAL PLOT SETTINGS
sns.set_theme(style="white", context="talk")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 14,
    'axes.labelsize': 18,
    'axes.labelweight': 'normal',
    'axes.titlesize': 20,
    'axes.titleweight': 'normal',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.max_open_warning': 0,
    'axes.spines.top': False,
    'axes.spines.right': False
})

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

FIBROSIS_ORDER = ['Uniform', 'Compact', 'Diffuse', 'Interstitial', 'Patchy']
TYPE_COLORS = {
    'Uniform': '#444444',
    'Compact': '#0000A2',
    'Diffuse': '#50AD9F',
    'Interstitial': '#E9C716',
    'Patchy': '#BC272D'
}

INVALID_NUMERIC = -9999.0 # Must match the simulator's flag

def apply_bold_ticks(ax):
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

# SEMANTIC METRIC CATEGORIZATION
def get_metric_categories(dim_mode):
    """Categorizes the curated metrics by their physical and mathematical meaning."""
    categories = {
        "Morphology_and_Geometry": {
            "norm_mean_cluster_size": r"Normalized Mean Cluster Size ($\hat{S}$)",
            "mean_aspect_ratio": "Mean Aspect Ratio"
        },
        "Texture_and_Complexity": {
            "morans_i": "Moran's I (Spatial Autocorrelation)",
            "fractal_dimension": r"Fractal Dimension ($D_f$)"
        },
        "Topology_and_Homology": {
            "betti_0": r"Betti 0 ($\beta_0$: Connected Components)",
            "betti_1": r"Betti 1 ($\beta_1$: 1D Loops)"
        },
        "Connectivity_and_Conduction": {
            "min_path_x": "Shortest Percolating Path Length",
            "topological_conduction_eff": "Topological Conduction Efficiency (TCE)"
        }
    }

    if dim_mode == "3D":
        categories["Topology_and_Homology"]["betti_2"] = r"Betti 2 ($\beta_2$: Enclosed Voids)"

    return categories

def get_all_metrics_flat(dim_mode):
    """Flattens the categorized dictionary into a single list of keys."""
    cats = get_metric_categories(dim_mode)
    return [metric for group in cats.values() for metric in group.keys()]

# MATHEMATICAL MODEL (CONSTRAINED SIGMOID)
def sigmoid(x, k, x0):
    """
    Constrained Sigmoid: Mathematically forced to pass exactly through (0,1) and (1,0).
    """
    def base_sig(t):
        return 1.0 / (1.0 + np.exp(np.clip(k * (t - x0), -500, 500)))
    
    f_x = base_sig(x)
    f_0 = base_sig(0.0) 
    f_1 = base_sig(1.0) 
    
    denom = f_0 - f_1
    if isinstance(denom, np.ndarray):
        denom = np.where(denom == 0, 1e-10, denom)
    else:
        denom = denom if denom != 0 else 1e-10
        
    return (f_x - f_1) / denom

def fit_sigmoid(x_data, y_data):
    """Fits the constrained sigmoid and computes the true exact threshold (true_pc)."""
    p0 = [20.0, 0.5]
    bounds = ([0.001, 0.0], [500.0, 1.0])
    try:
        popt, _ = curve_fit(sigmoid, x_data, y_data, p0=p0, bounds=bounds, maxfev=10000)
        k_opt, x0_opt = popt
        
        # Find exact density where probability is 0.5
        def objective(x):
            return sigmoid(x, k_opt, x0_opt) - 0.5
            
        sol = root_scalar(objective, bracket=[0.0, 1.0], method='brentq')
        true_pc = sol.root if sol.converged else x0_opt
        
        return k_opt, x0_opt, true_pc
    except (RuntimeError, ValueError):
        return None, None, None

def plot_percolation_and_get_thresholds(df, out_dir, dim_mode):
    """Fits sigmoids, extracts decay kinematics, and plots stratified by angle using tab10."""
    sub_dir = out_dir / "percolation_thresholds"
    sub_dir.mkdir(parents=True, exist_ok=True)
    
    thresholds = {} 
    decay_stats = {}
    combined_fit_data = []

    perc_col = 'percolates_x'
    if perc_col not in df.columns: return thresholds, decay_stats

    for ftype in [t for t in FIBROSIS_ORDER if t in df['fibrosis_type'].unique()]:
        df_type = df[df['fibrosis_type'] == ftype]
        if df_type.empty: continue
            
        color = TYPE_COLORS.get(ftype, '#000000')
        
        # 1. GLOBAL FIT (Against Density)
        glob_stats = df_type.groupby('density').agg(prob=(perc_col, 'mean')).reset_index()
        x_g, y_g = glob_stats['density'].values, glob_stats['prob'].values
        
        k_g, x0_g, pc_g = fit_sigmoid(x_g, y_g)
        
        if pc_g is not None:
            thresholds[ftype] = pc_g
            
            # CALCULATE RIGOROUS DECAY KINEMATICS
            transition_window = 4.3944 / abs(k_g) if k_g != 0 else np.nan
            max_slope = abs(k_g) / 4.0
            decay_stats[ftype] = {'window': transition_window, 'slope': max_slope}
            
            combined_fit_data.append({'type': ftype, 'k': k_g, 'x0': x0_g, 'pc': pc_g, 'x': x_g, 'y': y_g})
            
            # Global Plot
            fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
            ax.scatter(x_g, y_g, color='black', alpha=0.5, label='Empirical Mean')
            x_fit = np.linspace(0, 1, 300)
            
            ax.plot(x_fit, sigmoid(x_fit, k_g, x0_g), color=color, linewidth=2.5, label=f'Fit ($p_c={pc_g:.3f}$)')
            ax.axvline(x=pc_g, color='black', linestyle='--', alpha=0.6)
            
            ax.set_title(f'Global {dim_mode} Percolation Threshold: {ftype}', fontweight='bold')
            ax.set_xlabel('Fibrosis Density')
            ax.set_ylabel('Percolation Probability')
            apply_bold_ticks(ax)
            ax.legend()
            fig.savefig(sub_dir / f"sigmoid_global_{ftype.lower()}.png", bbox_inches='tight')
            plt.close(fig)

        # 2. STRATIFIED FIT (By angle)
        angles = sorted(df_type['fiber_angle_deg'].unique())
        if len(angles) > 1:
            fig_strat, ax_strat = plt.subplots(figsize=(10, 8), dpi=300)
            palette = sns.color_palette("tab10", len(angles))
            
            for ang, p_color in zip(angles, palette):
                df_ang = df_type[df_type['fiber_angle_deg'] == ang]
                ang_stats = df_ang.groupby('density')[perc_col].mean().reset_index()
                
                x_a, y_a = ang_stats['density'].values, ang_stats[perc_col].values
                k_a, x0_a, pc_a = fit_sigmoid(x_a, y_a)
                
                if pc_a is not None:
                    ax_strat.scatter(x_a, y_a, color=p_color, alpha=0.7, s=25)
                    x_fit = np.linspace(0, 1, 300)
                    ax_strat.plot(x_fit, sigmoid(x_fit, k_a, x0_a), color=p_color, linewidth=2.5, label=f"{int(ang)}° ($p_c={pc_a:.3f}$)")
                    
            ax_strat.set_title(f'{dim_mode} Percolation by Fiber Angle: {ftype}', fontweight='bold')
            ax_strat.set_xlabel('Fibrosis Density')
            ax_strat.set_ylabel('Percolation Probability')
            ax_strat.legend(title='Fiber Angle')
            apply_bold_ticks(ax_strat)
            plt.tight_layout()
            fig_strat.savefig(sub_dir / f"sigmoid_stratified_{ftype.lower()}.png", bbox_inches='tight')
            plt.close(fig_strat)

    if combined_fit_data:
        fig_comb, ax_comb = plt.subplots(figsize=(10, 8), dpi=300)
        x_fit = np.linspace(0, 1, 300)
        for data in combined_fit_data:
            ax_comb.plot(x_fit, sigmoid(x_fit, data['k'], data['x0']), color=TYPE_COLORS.get(data['type'], '#000'), linewidth=2.5, label=f"{data['type']} ($p_c={data['pc']:.3f}$)")
        
        ax_comb.set_title(f'{dim_mode} Percolation Thresholds Comparison', fontweight='bold')
        ax_comb.set_xlabel('Fibrosis Density')
        ax_comb.set_ylabel('Percolation Probability')
        ax_comb.legend()
        apply_bold_ticks(ax_comb)
        fig_comb.savefig(sub_dir / "sigmoid_combined.png", bbox_inches='tight')
        plt.close(fig_comb)

    return thresholds, decay_stats
    
# DATA PROCESSING
def load_and_clean_data(scenario_master_csv, min_den, max_den):
    """Loads CSVs, filters by density, handles duplicates, and maps INVALID_NUMERIC to NaN."""
    full_df = pd.read_csv(scenario_master_csv)
    
    if 'fibrosis_type' in full_df.columns:
        full_df['fibrosis_type'] = full_df['fibrosis_type'].str.capitalize()
    
    # Replace invalid/failed computation flags with NaN for clean statistics & plotting
    full_df = full_df.replace(INVALID_NUMERIC, np.nan)
    
    # Density filter
    full_df = full_df[(full_df['density'] >= min_den) & (full_df['density'] <= max_den)]
    
    if 'actual_density' not in full_df.columns:
        full_df['actual_density'] = full_df['density']
    
    # Drop duplicates
    initial_count = len(full_df)
    full_df = full_df.drop_duplicates(subset=['fibrosis_type', 'fiber_angle_deg', 'density', 'seed'], keep='last')
    if initial_count != len(full_df):
        print(f"Data Cleaning: Removed {initial_count - len(full_df)} duplicate records.")

    # Remove empty columns (example, grid_z in 2D, or metrics that all failed)
    full_df = full_df.dropna(axis=1, how='all')

    perc_col = 'percolates_x' 
    if perc_col in full_df.columns:
        full_df[perc_col] = full_df[perc_col].fillna(0).astype(int)

    return full_df

# SUMMARY TABLE EXPORT
def export_summary_tables(df, thresholds, decay_stats, out_dir, dim_mode):
    """Exports structured CSVs with means of valid values for table generation."""
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    
    avail_types = [t for t in FIBROSIS_ORDER if t in df['fibrosis_type'].unique()]
    
    # Table 1: Topology and Morphology
    topo_cols = ['morans_i', 'mean_aspect_ratio', 'fractal_dimension', 'betti_0', 'betti_1']
    if dim_mode == '3D': topo_cols.append('betti_2')
    topo_cols.append('norm_mean_cluster_size')
    
    topo_data = []
    for ftype in avail_types:
        row = {'Phenotype': ftype}
        df_type = df[df['fibrosis_type'] == ftype]
        for col in topo_cols:
            if col in df_type.columns:
                mean_val = df_type[col].dropna().mean()
                row[col] = mean_val
            else:
                row[col] = np.nan
        topo_data.append(row)
        
    df_topo = pd.DataFrame(topo_data)
    if 'norm_mean_cluster_size' in df_topo.columns:
        df_topo['norm_mean_cluster_size'] = df_topo['norm_mean_cluster_size'] * 100.0
        
    rename_dict = {
        'morans_i': 'I', 'mean_aspect_ratio': 'AR', 'fractal_dimension': 'D_f',
        'betti_0': 'beta_0', 'betti_1': 'beta_1', 'betti_2': 'beta_2',
        'norm_mean_cluster_size': 'S_hat_pct'
    }
    df_topo.rename(columns=rename_dict, inplace=True)
    df_topo.to_csv(table_dir / f"table_topology_{dim_mode.lower()}.csv", index=False, float_format="%.3f")
    
    # Table 2: Percolation
    perc_data = []
    for ftype in avail_types:
        row = {'Phenotype': ftype}
        row['p_c'] = thresholds.get(ftype, np.nan)
        row['p_tw'] = decay_stats.get(ftype, {}).get('window', np.nan)
        
        if 'topological_conduction_eff' in df.columns:
            row['TCE'] = df[df['fibrosis_type'] == ftype]['topological_conduction_eff'].dropna().mean()
        else:
            row['TCE'] = np.nan
        perc_data.append(row)
        
    df_perc = pd.DataFrame(perc_data)
    df_perc.to_csv(table_dir / f"table_percolation_{dim_mode.lower()}.csv", index=False, float_format="%.3f")
    print(f"Exported clean table data to {table_dir}")
    
# CORE PLOTTING MODULES
def plot_metric_vs_density(df, metric, title, ylabel, out_path):
    """Generates a lineplot using density on X-axis, automatically ignoring NaNs."""
    if metric not in df.columns or df[metric].dropna().empty: return
    
    # Drop NaNs specifically for this metric to ensure continuous lines
    df_plot = df[['fibrosis_type', 'density', metric]].dropna(subset=[metric])
    if df_plot.empty: return
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    sns.lineplot(data=df_plot, x='density', y=metric, hue='fibrosis_type',
                 palette=TYPE_COLORS, hue_order=[t for t in FIBROSIS_ORDER if t in df_plot['fibrosis_type'].unique()],
                 markers=False, dashes=False, errorbar=('ci', 95), linewidth=2.5, ax=ax)
            
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Fibrosis Density')
    ax.set_ylabel(ylabel)
    ax.legend(title=None)
    apply_bold_ticks(ax)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def generate_categorized_line_plots(df, out_dir, dim_mode):
    """Generates continuous plots sorted into semantic folders."""
    categories = get_metric_categories(dim_mode)
    
    for cat_name, metrics_dict in categories.items():
        cat_dir = out_dir / "line_plots" / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)
        
        for metric_col, metric_label in metrics_dict.items():
            out_path = cat_dir / f"line_{metric_col}.png"
            plot_metric_vs_density(df, metric_col, metric_label, metric_label, out_path)

def generate_distribution_plots(df, out_dir, dim_mode):
    """Boxplots to view variance and outliers, categorized."""
    categories = get_metric_categories(dim_mode)
    avail_types = [t for t in FIBROSIS_ORDER if t in df['fibrosis_type'].unique()]
    
    for cat_name, metrics_dict in categories.items():
        cat_dir = out_dir / "distributions" / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)
        
        for metric_col, metric_label in metrics_dict.items():
            if metric_col not in df.columns or df[metric_col].dropna().empty: continue
            clean_df = df.dropna(subset=[metric_col])
            
            fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
            sns.boxplot(data=clean_df, x='fibrosis_type', y=metric_col, palette=TYPE_COLORS, order=avail_types, ax=ax)
            ax.set_title(f"Variance & Outliers: {metric_label}", fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel(metric_label)
            apply_bold_ticks(ax)
            fig.savefig(cat_dir / f"box_{metric_col}.png", bbox_inches='tight')
            plt.close(fig)
            
def generate_summary_pointplots(df, out_dir, dim_mode):
    """Generates Pointplots (Mean + Std Dev error bars) categorized by metric."""
    categories = get_metric_categories(dim_mode)
    avail_types = [t for t in FIBROSIS_ORDER if t in df['fibrosis_type'].unique()]
    
    for cat_name, metrics_dict in categories.items():
        cat_dir = out_dir / "summary_pointplots" / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)
        
        for metric_col, metric_label in metrics_dict.items():
            if metric_col not in df.columns: continue
            
            metric_df = df[['fibrosis_type', metric_col]].dropna(subset=[metric_col])
            if metric_df.empty: continue
            
            fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
            
            # errorbar='sd'
            sns.pointplot(data=metric_df, x='fibrosis_type', y=metric_col, palette=TYPE_COLORS, errorbar='sd', 
                          order=avail_types, capsize=0.1, markers='o', linestyles='', ax=ax)
            
            ax.set_title(f"{metric_label} - Mean & Std Dev", fontweight='bold')
            ax.set_ylabel(metric_label)
            ax.set_xlabel('')
            apply_bold_ticks(ax)
            fig.savefig(cat_dir / f"pointplot_{metric_col}.png", bbox_inches='tight')
            plt.close(fig)

# STATISTICAL REPORT
def build_txt_report(df, thresholds, decay_stats, out_dir, dim_mode):
    report = []
    report.append("=========================================================")
    report.append(f"  FIBROSTRUCT {dim_mode}: COMPREHENSIVE STATISTICAL REPORT")
    report.append("=========================================================\n")
    
    report.append("GLOSSARY OF METRICS")
    report.append("Normalized Mean Cluster Size: The mass-weighted average size of finite obstacle clusters normalized by the total voxel volume. Excludes the largest cluster.")
    report.append("Mean Aspect Ratio: Bounding box geometric stretch (Length / Width) of the clusters weighted by their area.")
    report.append("Moran's I: Global spatial autocorrelation. +1.0 = highly clustered, 0.0 = random noise.")
    report.append("Fractal Dimension: Measures the complexity and space-filling capacity of the fibrosis boundaries.")
    report.append("Betti 0: Number of connected healthy tissue components.")
    report.append("Betti 1: Number of 1D healthy loops (fibrotic islands) inside the healthy tissue.")
    report.append("Betti 2 (3D only): Number of enclosed voids (fully enclosed fibrotic pockets) in the healthy tissue.")
    report.append("Topological Conduction Efficiency (TCE): The structural speed limit of the mesh (Linear Length / Average Percolating Path Length).")
    report.append("Shortest Percolating Path: The absolute minimal steps required to cross the domain X-axis.")
    report.append("\n")

    report.append(f"1. {dim_mode} PERCOLATION THRESHOLDS (pc) AND DECAY KINEMATICS")
    report.append("-" * 65)
    report.append("pc (Threshold): The exact density where probability of conduction hits 50%.")
    report.append("Transition Window: Density window required for the curve to drop from 90% to 10% connectivity (smaller = more abrupt transition).")
    report.append("\n")
    
    for ftype in FIBROSIS_ORDER:
        if ftype in thresholds and ftype in decay_stats:
            pc = thresholds[ftype]
            width = decay_stats[ftype]['window']
            slope = decay_stats[ftype]['slope']
            report.append(f"   {ftype.ljust(15)} : pc = {pc:.3f}  |  Transition Window (90% to 10%) = {width:.3f}  |  Max Slope = {slope:.3f}")
    report.append("\n")

    report.append("2. GENERAL DESCRIPTIVE STATISTICS PER METRIC")
    report.append("-" * 65)
    report.append("Aggregated mathematical overview comparing fibrosis types for each structural metric (Excludes invalid/failed samples).\n")
    
    metrics_flat = get_all_metrics_flat(dim_mode)
    avail_metrics = [m for m in metrics_flat if m in df.columns]
    
    cats = get_metric_categories(dim_mode)
    metric_labels = {}
    for cat_name, m_dict in cats.items():
        metric_labels.update(m_dict)
        
    avail_types = [t for t in FIBROSIS_ORDER if t in df['fibrosis_type'].unique()]
    
    for m in avail_metrics:
        m_label = metric_labels.get(m, m)
        report.append(f"\n>> {m_label.upper()} ({m})")
        
        for ftype in avail_types:
            clean_series = df[df['fibrosis_type'] == ftype][m].dropna()
            if clean_series.empty: continue
                
            m_mean = clean_series.mean()
            m_median = clean_series.median()
            m_std = clean_series.std()
            
            report.append(f"   {ftype.ljust(15)} | Mean: {m_mean:>12.3f} | Std: {m_std:>12.3f} | Valid N: {len(clean_series)}")

    report.append("\n")
    
    report.append("3. STATISTICAL SIGNIFICANCE (Kruskal-Wallis H-test across Fibrosis Types)")
    report.append("-" * 65)
    report.append("Null hypothesis: Medians of all groups are equal (alpha = 0.05).")
    report.append("If p-value < 0.05, the fibrosis phenotype significantly alters the metric.\n")
    
    for metric in metrics_flat:
        if metric not in df.columns: continue
        groups = [df[df['fibrosis_type'] == ftype][metric].dropna().values 
                  for ftype in FIBROSIS_ORDER if ftype in df['fibrosis_type'].unique()]
        valid_groups = [g for g in groups if len(g) > 0]
        
        if len(valid_groups) > 1:
            stat, p_val = stats.kruskal(*valid_groups)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            report.append(f"{metric.ljust(30)} | H-stat: {stat:>12.2f} | p-value: {p_val:>12.2e} [{sig}]")
    report.append("\n")

    report_path = out_dir / f"structure_report.txt"
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("\n".join(report))

# MAIN EXECUTOR
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_dir", type=str, required=True, help="Base directory for storing configurations, outputs, and results.")
    parser.add_argument("--dim", type=str, required=True, choices=["2D", "3D"], help="Simulation dimension.")
    parser.add_argument("--shape", type=str, required=True, help="Geometric shape for fibrosis distribution (e.g., 'ellipse', 'box', 'full').")
    parser.add_argument("--min_den", type=float, default=0.0, help="Minimum density to plot.")
    parser.add_argument("--max_den", type=float, default=1.0, help="Maximum density to plot.")
    
    args = parser.parse_args()
    
    scenario_batch_dir = PROJECT_ROOT / args.batch_dir / args.dim / args.shape
    if not scenario_batch_dir.exists():
        print(f"Error: Scenario batch directory not found: {scenario_batch_dir}")
        print("Please ensure the orchestrator.py completed successfully and created the expected directory structure.")
        return
    
    scenario_master_csv = scenario_batch_dir / "data" / "master_dataset.csv"
    if not scenario_master_csv.exists():
        print(f"Error: master_dataset.csv not found in expected location: {scenario_master_csv}")
        print("Please ensure the orchestrator.py completed successfully and created the expected directory structure.")
        return
    
    try:
        df = load_and_clean_data(scenario_master_csv, args.min_den, args.max_den)
    except Exception as e:
        print(f"Error loading data from {scenario_master_csv}: {e}")
        return
        
    if df.empty:
        print(f"Error: The filtered dataset from {scenario_master_csv} is empty.")
        return

    out_path = scenario_batch_dir / "analysis" / "structure"
    out_path.mkdir(parents=True, exist_ok=True)
    
    processed_csv_path = out_path / f"processed_dataset.csv"
    df.to_csv(processed_csv_path, index=False)
    print(f"Processed dataset saved: {processed_csv_path}")

    print("Fitting Sigmoids and extracting Thresholds & Decay Stats")
    thresholds, decay_stats = plot_percolation_and_get_thresholds(df, out_path, args.dim)
    
    print("Generating Summary Tables (CSV format for external use)")
    export_summary_tables(df, thresholds, decay_stats, out_path, args.dim)
    
    print("Generating Semantic Line Plots")
    generate_categorized_line_plots(df, out_path, args.dim)
    
    print("Generating Categorized Boxplots")
    generate_distribution_plots(df, out_path, args.dim)
    
    print("Generating Summary Pointplots (Mean & Std Dev)")
    generate_summary_pointplots(df, out_path, args.dim)
    
    print("Running Non-Parametric Statistical Tests & Descriptive Stats")
    build_txt_report(df, thresholds, decay_stats, out_path, args.dim)
    
    print(f"SUCCESS! All categorized analyses saved to: {out_path}")

if __name__ == "__main__":
    main()