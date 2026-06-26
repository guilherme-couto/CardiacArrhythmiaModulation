"""
Multivariate Analysis Module for FibroStruct (2D and 3D).

Executes Principal Component Analysis (PCA), PCA Biplots, Correlation Heatmaps,
KDE distributions, Probability curves, and Density-based lineplots.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import math
import sys
import warnings

warnings.filterwarnings('ignore')

# GLOBAL PLOT SETTINGS
sns.set_theme(style="white", context="talk")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 20,
    'axes.labelsize': 20,
    'axes.labelweight': 'normal',
    'axes.titlesize': 20,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'figure.max_open_warning': 0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.fontsize': 20
})

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INVALID_NUMERIC = -9999.0

FIBROSIS_ORDER = ['Uniform', 'Compact', 'Diffuse', 'Interstitial', 'Patchy']
TYPE_COLORS = {
    'Uniform': '#444444',
    'Compact': '#0000A2',
    'Diffuse': '#50AD9F',
    'Interstitial': '#E9C716',
    'Patchy': '#BC272D'
}

OUTCOME_LABELS = {0: 'No Sustained Reentry', 1: 'Sustained Reentry', False: 'No Sustained Reentry', True: 'Sustained Reentry'}
OUTCOME_COLORS = {'No Sustained Reentry': '#1f77b4', 'Sustained Reentry': '#d62728'} 

# Labels for plots
METRIC_LABELS = {
    'norm_mean_cluster_size': 'Normalized Mean Cluster Size',
    'mean_aspect_ratio': 'Mean Aspect Ratio',
    'fractal_dimension': 'Fractal Dimension',
    'morans_i': "Moran's I",
    'betti_0': 'Betti 0',
    'betti_1': 'Betti 1',
    'betti_2': 'Betti 2',
    'topological_conduction_eff': 'Topological Conduction Efficiency',
    'min_path_x': 'Shortest Percolating Path',
}

METRIC_SHORTS = {
    'norm_mean_cluster_size': r'$\mathbf{\hat{S}}$',
    'mean_aspect_ratio': 'AR',
    'fractal_dimension': r'$\mathbf{D_f}$',
    'morans_i': r"$\mathbf{I}$",
    'betti_0': r'$\mathbf{\beta_0}$',
    'betti_1': r'$\mathbf{\beta_1}$',
    'betti_2': r'$\mathbf{\beta_2}$',
    'topological_conduction_eff': 'TCE',
    'min_path_x': 'Min Path',
}

def apply_bold_ticks(ax):
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

def clean_name(metric):
    return METRIC_LABELS.get(metric, metric.replace('_', ' ').title())

def short_name(metric):
    return METRIC_SHORTS.get(metric, metric.replace('_', ' ').title())

def get_scenario_name(dim, shape):
    mapping = {
        ('2D', 'full'): 'Scenario I',
        ('2D', 'ellipse'): 'Scenario II',
        ('3D', 'full'): 'Scenario III',
        ('3D', 'ellipse'): 'Scenario IV'
    }
    return mapping.get((dim, shape), f"Scenario ({dim}-{shape})")

def get_structural_metrics(df, custom_metrics=None):
    if custom_metrics:
        valid_metrics = [m for m in custom_metrics if m in df.columns]
        if not valid_metrics:
            raise ValueError("None of the specified metrics were found in the dataset.")
        return valid_metrics
        
    exclude = ['seed', 'actual_density', 'density', 'fiber_angle_deg', 'grid_x', 'grid_y', 'grid_z', 
               'dx', 'shape', 'atpi', 'homog_factor', 'final_time_ms', 'sustained_reentry', 'percolates_x']
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)
    if 'fibrosis_type' in df.columns:
        df['fibrosis_type'] = df['fibrosis_type'].str.strip().str.capitalize()
    
    df = df.replace(INVALID_NUMERIC, np.nan)
    df = df.dropna(axis=1, how='all')
    if 'sustained_reentry' in df.columns:
        df['sustained_reentry'] = df['sustained_reentry'].fillna(0).astype(int)
    return df

# PCA ANALYSIS & BIPLOT
def perform_pca_analysis(df, metrics, out_dir, scenario_name):
    pca_dir = out_dir / "pca_space"
    pca_dir.mkdir(parents=True, exist_ok=True)
    
    pca_df = df.dropna(subset=metrics)
    if len(pca_df) < 10:
        print("Not enough complete samples to perform PCA.")
        return None, None

    features = pca_df[metrics].values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    var_explained = pca.explained_variance_ratio_
    
    pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pc_df['Phenotype'] = pca_df['fibrosis_type'].values
    
    # Safe outcome mapping
    pc_df['Reentry'] = pca_df['sustained_reentry'].map(OUTCOME_LABELS).values
    pc_df['Raw_Reentry'] = pca_df['sustained_reentry'].values 
    
    # Plot 1: PCA by Phenotype
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    sns.scatterplot(data=pc_df, x='PC1', y='PC2', hue='Phenotype', palette=TYPE_COLORS, 
                    alpha=0.7, s=70, edgecolor='w', ax=ax)
    ax.set_title(f"{scenario_name} - PCA: Structural Phenotype Clustering", fontsize=22, fontweight='bold')
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% Variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% Variance)")
    ax.legend(title=None)
    apply_bold_ticks(ax)
    fig.savefig(pca_dir / "pca_clustering_phenotype.png", bbox_inches='tight')
    plt.close(fig)

    # Plot 2: PCA Biplot
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    sns.scatterplot(data=pc_df, x='PC1', y='PC2', hue='Phenotype', palette=TYPE_COLORS, 
                    alpha=0.5, s=60, edgecolor='none', ax=ax)
    
    coeff = pca.components_.T
    scale_x = np.max(np.abs(pc_df['PC1'])) * 0.8 / np.max(np.abs(coeff[:, 0])) if np.max(np.abs(coeff[:, 0])) != 0 else 1
    scale_y = np.max(np.abs(pc_df['PC2'])) * 0.8 / np.max(np.abs(coeff[:, 1])) if np.max(np.abs(coeff[:, 1])) != 0 else 1
    
    for i in range(len(metrics)):
        ax.arrow(0, 0, coeff[i,0]*scale_x, coeff[i,1]*scale_y, color='k', alpha=0.8, head_width=0.2, head_length=0.2, linewidth=1.5)
        ax.text(coeff[i,0]*scale_x*1.15, coeff[i,1]*scale_y*1.15, short_name(metrics[i]), 
                color='k', ha='center', va='center', fontweight='bold', fontsize=14, 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                
    ax.set_title(f"{scenario_name} - PCA Biplot: Feature Vectors", fontsize=22, fontweight='bold')
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% Variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% Variance)")
    ax.legend(title=None, loc='best', fontsize=16)
    apply_bold_ticks(ax)
    fig.savefig(pca_dir / "pca_biplot_phenotype.png", bbox_inches='tight')
    plt.close(fig)

    # Plot 3: PCA by Sustained Reentry
    pc_df_sorted = pc_df.sort_values(by='Raw_Reentry', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    sns.scatterplot(data=pc_df_sorted, x='PC1', y='PC2', hue='Reentry', 
                    palette=OUTCOME_COLORS, alpha=0.8, s=70, edgecolor='w', ax=ax)
    ax.set_title(f"{scenario_name} - PCA: Structural Vulnerability Space", fontsize=22, fontweight='bold')
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% Variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% Variance)")
    ax.legend(title=None)
    apply_bold_ticks(ax)
    fig.savefig(pca_dir / "pca_clustering_reentry.png", bbox_inches='tight')
    plt.close(fig)
    
    clean_metrics = [clean_name(m) for m in metrics]
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=clean_metrics)
    return pca, loadings

# HEATMAPS CORRELATION
def export_and_plot_correlations(df, metrics, out_dir, dim_mode):
    corr_dir = out_dir / "correlation_matrices"
    corr_dir.mkdir(parents=True, exist_ok=True)
    
    short_metrics = [short_name(m) for m in metrics]
    
    corr_spearman = df[metrics].corr(method='spearman')
    corr_pearson = df[metrics].corr(method='pearson')
    
    corr_spearman.columns = short_metrics; corr_spearman.index = short_metrics
    corr_pearson.columns = short_metrics; corr_pearson.index = short_metrics

    cmap = sns.color_palette("vlag", as_cmap=True)
    mask = np.triu(np.ones_like(corr_spearman, dtype=bool))

    # Standard Side-by-Side Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(28, 12), dpi=300)
    
    sns.heatmap(corr_spearman, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=1.5, annot=True, fmt=".2f", 
                annot_kws={"size": 14, "weight": "bold"}, cbar=False, ax=axes[0])
    axes[0].set_title(f"Spearman Correlation", fontsize=26, fontweight='bold', pad=20)
    axes[0].tick_params(axis='both', which='both', length=0)
    apply_bold_ticks(axes[0])
    
    sns.heatmap(corr_pearson, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=1.5, annot=True, fmt=".2f", 
                annot_kws={"size": 14, "weight": "bold"}, cbar=False, ax=axes[1])
    axes[1].set_title(f"Pearson Correlation", fontsize=26, fontweight='bold', pad=20)
    axes[1].tick_params(axis='both', which='both', length=0)
    apply_bold_ticks(axes[1])
    
    plt.tight_layout()
    fig.savefig(corr_dir / "heatmap_standard_combined.png", bbox_inches='tight')
    plt.close(fig)

    # Flatten Data for CSV
    def get_flattened_stats(target_df):
        c_s = target_df[metrics].corr(method='spearman')
        c_p = target_df[metrics].corr(method='pearson')
        cov_m = target_df[metrics].cov()
        m_triu = np.triu(np.ones(c_s.shape), k=1).astype(bool)
        
        flat_s = c_s.where(m_triu).unstack().dropna().reset_index()
        flat_s.columns = ['Metric_1', 'Metric_2', 'Spearman_Rho']
        flat_p = c_p.where(m_triu).unstack().dropna().reset_index()
        flat_p.columns = ['Metric_1', 'Metric_2', 'Pearson_R']
        flat_cov = cov_m.where(m_triu).unstack().dropna().reset_index()
        flat_cov.columns = ['Metric_1', 'Metric_2', 'Covariance']
        
        merged = pd.merge(flat_s, flat_p, on=['Metric_1', 'Metric_2'])
        merged = pd.merge(merged, flat_cov, on=['Metric_1', 'Metric_2'])
        merged['Abs_Spearman'] = merged['Spearman_Rho'].abs()
        return merged.sort_values(by='Abs_Spearman', ascending=False).drop(columns=['Abs_Spearman'])

    all_stats = [get_flattened_stats(df).assign(Scope='Global')]
    for ftype in [t for t in FIBROSIS_ORDER if t in df['fibrosis_type'].unique()]:
        df_type = df[df['fibrosis_type'] == ftype]
        if not df_type.empty:
            all_stats.append(get_flattened_stats(df_type).assign(Scope=ftype))

    flattened_df = pd.concat(all_stats, ignore_index=True)
    flattened_df = flattened_df[['Scope', 'Metric_1', 'Metric_2', 'Spearman_Rho', 'Pearson_R', 'Covariance']]
    
    csv_path = corr_dir / f"full_correlation_stats_{dim_mode.lower()}.csv"
    flattened_df.to_csv(csv_path, index=False, float_format='%.6f')
    
    return flattened_df

def create_txt_report(df, metrics, pca, loadings, flattened_corr_df, out_dir, scenario_name):
    report_path = out_dir / "multivariate_detailed_report.txt"
    
    lines = ["======================================================================================",
             f" MULTIVARIATE ANALYSIS DETAILED REPORT - {scenario_name.upper()}",
             "======================================================================================\n"]
    
    if pca is not None and loadings is not None:
        var = pca.explained_variance_ratio_
        lines.extend(["PRINCIPAL COMPONENT ANALYSIS (PCA)",
                      f"Total variance explained by PC1 and PC2: {(var[0]+var[1])*100:.2f}%\n"])
        for idx, pc in enumerate(['PC1', 'PC2']):
            lines.append(f">> {pc} Contributions (Explains {var[idx]*100:.2f}% of variance):")
            sorted_loadings = loadings[pc].reindex(loadings[pc].abs().sort_values(ascending=False).index)
            for metric, val in sorted_loadings.items():
                lines.append(f"   {metric.ljust(35)} : {val:+.4f}")
            lines.append("")
            
    lines.extend(["STRONG COLLINEARITIES (|r| > 0.9)",
                  "Variables showing redundant physical behaviors.\n"])
    scopes = flattened_corr_df['Scope'].unique()
    for scope in scopes:
        scope_data = flattened_corr_df[flattened_corr_df['Scope'] == scope]
        strong_spearman = scope_data[scope_data['Spearman_Rho'].abs() > 0.9]
        strong_pearson = scope_data[scope_data['Pearson_R'].abs() > 0.9]
        
        lines.append(f">> SCOPE: {scope.upper()}")
        lines.append("   [Spearman |rho| > 0.9]")
        if strong_spearman.empty: lines.append("      None")
        for _, row in strong_spearman.iterrows():
            lines.append(f"      {row['Spearman_Rho']:+.3f} : {clean_name(row['Metric_1'])} <--> {clean_name(row['Metric_2'])}")
            
        lines.append("   [Pearson |r| > 0.9]")
        if strong_pearson.empty: lines.append("      None")
        for _, row in strong_pearson.iterrows():
            lines.append(f"      {row['Pearson_R']:+.3f} : {clean_name(row['Metric_1'])} <--> {clean_name(row['Metric_2'])}")
        lines.append("")

    lines.extend(["POINT-BISERIAL CORRELATION (UNIVARIATE)",
                  "Evaluates the predictive strength of individual structural metrics against Sustained Reentry.",
                  "Interpretation of the r_pb coefficient:",
                  " - Positive (+) values : Higher metric values increase the chance of reentry (Risk factor).",
                  " - Negative (-) values : Higher metric values decrease the chance of reentry (Protective factor).",
                  " - Magnitude (|r_pb|)  : Values closer to 1.0 or -1.0 indicate strong separation between outcomes.\n"])
                  
    df_pb = df.copy()
    pb_results = []
    for m in metrics:
        clean_df = df_pb[[m, 'sustained_reentry']].dropna()
        if len(clean_df) > 10:
            r_pb, p_val = stats.pointbiserialr(clean_df[m], clean_df['sustained_reentry'])
            pb_results.append((clean_name(m), r_pb, p_val))
            
    pb_results.sort(key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True)
    for m_name, r_pb, p_val in pb_results:
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        lines.append(f"   {m_name.ljust(35)} : r_pb = {r_pb:>6.3f}  |  p-val = {p_val:>8.2e} [{sig}]")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

# OUTCOME ANALYSIS (REENTRY)
def analyze_reentry_boxplots(df, metrics, out_dir, scenario_name):
    re_dir = out_dir / "reentry_boxplots" 
    re_dir.mkdir(parents=True, exist_ok=True)
    avail_types = [t for t in FIBROSIS_ORDER if t in df['fibrosis_type'].unique()]
    
    df_plot = df.copy()
    df_plot['Outcome_Label'] = df_plot['sustained_reentry'].map(OUTCOME_LABELS)
    
    for metric in metrics:
        df_metric = df_plot.copy()
        
        # Ensures every group combination exists before plotting
        for ftype in avail_types:
            for outcome in ['No Sustained Reentry', 'Sustained Reentry']:
                if not ((df_metric['fibrosis_type'] == ftype) & (df_metric['Outcome_Label'] == outcome)).any():
                    dummy = pd.DataFrame([{'fibrosis_type': ftype, 'Outcome_Label': outcome, metric: np.nan}])
                    df_metric = pd.concat([df_metric, dummy], ignore_index=True)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        sns.boxplot(data=df_metric, x='fibrosis_type', y=metric, hue='Outcome_Label',
                    palette=OUTCOME_COLORS, order=avail_types, ax=ax)
        
        ax.set_title(f"{scenario_name}: Impact of {clean_name(metric)} on Reentry", fontsize=18, fontweight='bold')
        ax.set_xlabel("Fibrosis Type")
        ax.set_ylabel(clean_name(metric))
        ax.legend(title=None)
        apply_bold_ticks(ax)
        fig.savefig(re_dir / f"boxplot_reentry_{metric}.png", bbox_inches='tight')
        plt.close(fig)

def analyze_reentry_kde(df, metrics, out_dir, scenario_name):
    kde_dir = out_dir / "reentry_kde_maps" 
    kde_dir.mkdir(parents=True, exist_ok=True)
    
    df_plot = df.copy()
    df_plot['Outcome_Label'] = df_plot['sustained_reentry'].map(OUTCOME_LABELS)
    avail_types = [t for t in FIBROSIS_ORDER if t in df_plot['fibrosis_type'].unique()]
    
    cols = 3
    rows = math.ceil(len(metrics) / cols)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 3))
    
    # 2A. Combined KDE by Outcome
    fig, axes = plt.subplots(rows, cols, figsize=(cols*8, rows*7), dpi=300)
    axes = axes.flatten()
    for i, metric in enumerate(metrics):
        if df_plot[metric].dropna().empty: continue
        sns.histplot(data=df_plot, x=metric, hue='Outcome_Label', stat="density", common_norm=False, 
                     kde=True, bins=30, alpha=0.4, palette=OUTCOME_COLORS, ax=axes[i], legend=(i==0))
        axes[i].set_title(clean_name(metric), fontweight='bold', pad=15)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Probability Density" if i % cols == 0 else "")
        axes[i].xaxis.set_major_formatter(formatter)
        axes[i].yaxis.set_major_formatter(formatter)
        fig.canvas.draw_idle()
        apply_bold_ticks(axes[i])
    
    for j in range(i + 1, len(axes)): axes[j].axis('off')
    plt.subplots_adjust(wspace=0.4, hspace=0.5, bottom=0.15) 
    
    if axes[0].get_legend() is not None:
        fig.legend(handles=axes[0].get_legend().legend_handles, labels=list(OUTCOME_LABELS.values()), 
                   loc="lower center", ncol=2, frameon=True, fontsize=20, bbox_to_anchor=(0.5, 0.06))
        axes[0].get_legend().remove()
    
    fig.suptitle(f"{scenario_name} - Combined KDE Reentry Maps by Outcome", fontsize=26, fontweight='bold')
    fig.savefig(kde_dir / "kde_combined_by_outcome.png", bbox_inches='tight')
    plt.close(fig)
    
    # 2B. Combined KDE by Phenotype
    fig, axes = plt.subplots(rows, cols, figsize=(cols*8, rows*7), dpi=300)
    axes = axes.flatten()
    for i, metric in enumerate(metrics):
        if df_plot[metric].dropna().empty: continue
        sns.histplot(data=df_plot, x=metric, hue='fibrosis_type', stat="density", common_norm=False, 
                     kde=True, bins=30, alpha=0.3, palette=TYPE_COLORS, hue_order=avail_types, ax=axes[i], legend=(i==0))
        axes[i].set_title(clean_name(metric), fontweight='bold', pad=15)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Probability Density" if i % cols == 0 else "")
        axes[i].xaxis.set_major_formatter(formatter)
        axes[i].yaxis.set_major_formatter(formatter)
        fig.canvas.draw_idle() 
        apply_bold_ticks(axes[i])
        
    for j in range(i + 1, len(axes)): axes[j].axis('off')
    plt.subplots_adjust(wspace=0.4, hspace=0.5, bottom=0.15)
    
    if axes[0].get_legend() is not None:
        fig.legend(handles=axes[0].get_legend().legend_handles, labels=avail_types, 
                   loc="lower center", ncol=5, frameon=True, fontsize=20, bbox_to_anchor=(0.5, 0.06))
        axes[0].get_legend().remove()
        
    fig.suptitle(f"{scenario_name} - Combined KDE Reentry Maps by Phenotype", fontsize=26, fontweight='bold')
    fig.savefig(kde_dir / "kde_combined_by_phenotype.png", bbox_inches='tight')
    plt.close(fig)

def analyze_density_lineplots(df, metrics, out_dir, scenario_name):
    line_dir = out_dir / "density_trends"
    line_dir.mkdir(parents=True, exist_ok=True)
    
    df_plot = df.copy()
    df_plot['Outcome_Label'] = df_plot['sustained_reentry'].map(OUTCOME_LABELS)
    avail_types = [t for t in FIBROSIS_ORDER if t in df_plot['fibrosis_type'].unique()]
    
    cols = 3
    rows = math.ceil(len(metrics) / cols)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 3))
    
    # Combined Lineplots by Phenotype
    fig, axes = plt.subplots(rows, cols, figsize=(cols*8, rows*8), dpi=300)
    axes = axes.flatten()
    for i, metric in enumerate(metrics):
        if df_plot[metric].dropna().empty: continue
        sns.lineplot(data=df_plot, x='density', y=metric, hue='fibrosis_type', palette=TYPE_COLORS, 
                     hue_order=avail_types, errorbar=('ci', 95), ax=axes[i], legend=(i==0))
        axes[i].set_title(clean_name(metric), fontweight='bold', pad=15)
        axes[i].set_xlabel("Fibrosis Density")
        axes[i].set_ylabel("")
        axes[i].xaxis.set_major_formatter(formatter)
        axes[i].yaxis.set_major_formatter(formatter)
        fig.canvas.draw_idle() 
        apply_bold_ticks(axes[i])
        
    for j in range(i + 1, len(axes)): axes[j].axis('off')
    plt.subplots_adjust(wspace=0.4, hspace=0.5, bottom=0.15)
    
    if axes[0].get_legend() is not None:
        fig.legend(handles=axes[0].get_legend().legend_handles, labels=avail_types, 
                   loc="lower center", ncol=5, frameon=True, fontsize=20, bbox_to_anchor=(0.5, 0.06), markerscale=2)
        axes[0].get_legend().remove()
        
    fig.suptitle(f"{scenario_name} - Structural Metrics vs Density by Phenotype", fontsize=26, fontweight='bold')
    fig.savefig(line_dir / "lineplots_combined_by_phenotype.png", bbox_inches='tight')
    plt.close(fig)
    
    # Combined Lineplots by Outcome
    fig, axes = plt.subplots(rows, cols, figsize=(cols*8, rows*7), dpi=300)
    axes = axes.flatten()
    for i, metric in enumerate(metrics):
        if df_plot[metric].dropna().empty: continue
        sns.lineplot(data=df_plot, x='density', y=metric, hue='Outcome_Label', palette=OUTCOME_COLORS, 
                     errorbar=('ci', 95), ax=axes[i], legend=(i==0))
        axes[i].set_title(clean_name(metric), fontweight='bold', pad=15)
        axes[i].set_xlabel("Fibrosis Density")
        axes[i].set_ylabel("")
        axes[i].xaxis.set_major_formatter(formatter)
        axes[i].yaxis.set_major_formatter(formatter)
        fig.canvas.draw_idle() 
        apply_bold_ticks(axes[i])
        
    for j in range(i + 1, len(axes)): axes[j].axis('off')
    plt.subplots_adjust(wspace=0.4, hspace=0.5, bottom=0.15)
    
    if axes[0].get_legend() is not None:
        fig.legend(handles=axes[0].get_legend().legend_handles, labels=list(OUTCOME_LABELS.values()), 
                   loc="lower center", ncol=2, frameon=True, fontsize=20, bbox_to_anchor=(0.5, 0.06), markerscale=2)
        axes[0].get_legend().remove()
        
    fig.suptitle(f"{scenario_name} - Structural Metrics vs Density by Outcome", fontsize=26, fontweight='bold')
    fig.savefig(line_dir / "lineplots_combined_by_outcome.png", bbox_inches='tight')
    plt.close(fig)

def analyze_reentry_probability_curves(df, metrics, out_dir, scenario_name):
    prob_dir = out_dir / "reentry_probabilities"
    prob_dir.mkdir(parents=True, exist_ok=True)
    
    df_plot = df.copy()
    avail_types = [t for t in FIBROSIS_ORDER if t in df_plot['fibrosis_type'].unique()]
    
    for metric in metrics:
        if df_plot[metric].dropna().empty: continue
            
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        min_val, max_val = df_plot[metric].min(), df_plot[metric].max()
        if min_val == max_val: continue 
            
        bins = np.linspace(min_val, max_val, 20)
        df_plot['metric_binned'] = pd.cut(df_plot[metric], bins=bins)
        df_plot['bin_mid'] = df_plot['metric_binned'].apply(lambda x: x.mid if pd.notnull(x) else np.nan).astype(float)
        
        sns.lineplot(data=df_plot, x='bin_mid', y='sustained_reentry', 
                     hue='fibrosis_type', palette=TYPE_COLORS, hue_order=avail_types,
                     marker='o', markersize=8, linewidth=2.5, errorbar=None, ax=ax)
        
        ax.set_title(f"{scenario_name} - Prob. of Reentry vs {clean_name(metric)}", fontweight='bold')
        ax.set_xlabel(clean_name(metric))
        ax.set_ylabel("Probability of Sustained Reentry")
        
        max_prob = df_plot.groupby(['fibrosis_type', 'metric_binned'], observed=True)['sustained_reentry'].mean().max()
        if not np.isnan(max_prob):
            ax.set_ylim(-0.02, min(1.05, max_prob * 1.15)) 
            
        ax.legend(title=None)
        apply_bold_ticks(ax)
        fig.savefig(prob_dir / f"prob_curve_{metric}.png", bbox_inches='tight')
        plt.close(fig)

# PAIRPLOTS
def generate_pairplots(df, metrics, out_dir, scenario_name):
    pair_dir = out_dir / "bivariate_pairplots"
    pair_dir.mkdir(parents=True, exist_ok=True)
    
    if len(metrics) > 12:
        print("Warning: Pairplot generated with >12 metrics. It may be dense and slow to render.")
        
    df_plot = df.copy()
    df_plot = df_plot.rename(columns={m: short_name(m) for m in metrics})
    short_metrics = [short_name(m) for m in metrics]
    
    def clean_upper_triangle(x, y, **kws):
        ax=plt.gca()
        ax.axis('off')
        
    avail_types = [t for t in FIBROSIS_ORDER if t in df_plot['fibrosis_type'].unique()]

    print("   Drawing Phenotype Pairplot...")
    g_pheno = sns.PairGrid(df_plot, vars=short_metrics, hue="fibrosis_type", palette=TYPE_COLORS, diag_sharey=False, hue_order=avail_types)
    g_pheno.map_lower(sns.scatterplot, alpha=0.6, s=15, edgecolor='none')
    g_pheno.map_diag(sns.kdeplot, fill=True, alpha=0.5, warn_singular=False)
    g_pheno.map_upper(clean_upper_triangle)
    
    for ax in g_pheno.axes.flatten():
        if ax is not None:
            ax.set_xlabel(ax.get_xlabel(), fontweight='bold')
            ax.set_ylabel(ax.get_ylabel(), fontweight='bold')
            
    g_pheno.add_legend(title="")
    sns.move_legend(g_pheno, "upper right", bbox_to_anchor=(0.8, 0.8), ncol=1, title=None, frameon=True, markerscale=4, fontsize=22)
    
    g_pheno.fig.suptitle(f"{scenario_name} - Bivariate Structural Metrics by Phenotype", fontsize=26, fontweight='bold')
    g_pheno.savefig(pair_dir / "pairplot_by_phenotype.png", bbox_inches='tight')
    plt.close(g_pheno.fig)

    print("   Drawing Reentry Pairplot...")
    df_plot['Outcome_Label'] = df_plot['sustained_reentry'].map(OUTCOME_LABELS)
    df_sorted = df_plot.sort_values(by='sustained_reentry', ascending=True)
    
    g_reentry = sns.PairGrid(df_sorted, vars=short_metrics, hue="Outcome_Label", palette=OUTCOME_COLORS, diag_sharey=False)
    g_reentry.map_lower(sns.scatterplot, alpha=0.7, s=20, edgecolor='w')
    g_reentry.map_diag(sns.kdeplot, fill=True, alpha=0.5, warn_singular=False, common_norm=False)
    g_reentry.map_upper(clean_upper_triangle)
    
    for ax in g_reentry.axes.flatten():
        if ax is not None:
            ax.set_xlabel(ax.get_xlabel(), fontweight='bold')
            ax.set_ylabel(ax.get_ylabel(), fontweight='bold')

    g_reentry.add_legend(title="")
    sns.move_legend(g_reentry, "upper right", bbox_to_anchor=(0.8, 0.8), ncol=1, title=None, frameon=True, markerscale=4, fontsize=22)
    
    g_reentry.fig.suptitle(f"{scenario_name} - Bivariate Structural Metrics by Reentry Outcome", fontsize=26, fontweight='bold')
    g_reentry.savefig(pair_dir / "pairplot_by_reentry.png", bbox_inches='tight')
    plt.close(g_reentry.fig)


# EXECUTOR
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_dir", type=str, required=True, help="Base directory for storing configurations, outputs, and results.")
    parser.add_argument("--dim", type=str, required=True, choices=["2D", "3D"], help="Simulation dimension.")
    parser.add_argument("--shape", type=str, required=True, help="Geometric shape for fibrosis distribution (e.g., 'ellipse', 'box', 'full').")
    parser.add_argument("--metrics", type=str, nargs='+', help="Specific metrics to analyze (Space-separated).")
    
    args = parser.parse_args()
    scenario_name = get_scenario_name(args.dim, args.shape)
    
    scenario_batch_dir = PROJECT_ROOT / args.batch_dir / args.dim / args.shape
    scenario_master_csv = scenario_batch_dir / "data" / "master_dataset.csv"
    
    if not scenario_master_csv.exists():
        print(f"\n[!] FATAL ERROR: master_dataset.csv not found at {scenario_master_csv}")
        sys.exit(1)
        
    out_path = scenario_batch_dir / "analysis" / "multivariate"
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset for {scenario_name}...")
    df = load_and_clean_data(scenario_master_csv)
    
    if 'sustained_reentry' not in df.columns:
        print("\n[!] FATAL ERROR: 'sustained_reentry' column not found in the CSV.")
        print("    This module requires outcome data. Aborting execution.")
        sys.exit(1)
        
    metrics = get_structural_metrics(df, custom_metrics=args.metrics)
    if not metrics:
        print("\n[!] FATAL ERROR: No valid continuous structural metrics found to correlate.")
        sys.exit(1)
        
    print(f"Analyzing {len(metrics)} structural metrics...")

    print("1. Executing PCA Analysis...")
    pca, loadings = perform_pca_analysis(df, metrics, out_path, scenario_name)
    
    print("2. Computing Correlation Heatmaps (Pearson & Spearman)...")
    structural_only_metrics = [m for m in metrics if m not in ['min_path_x', 'topological_conduction_eff', 'percolates_x', 'percolates_horiz', 'percolates_y', 'percolates_z']]
    flattened_corr_df = export_and_plot_correlations(df, structural_only_metrics, out_path, args.dim)
    
    # Text Report Generation
    create_txt_report(df, metrics, pca, loadings, flattened_corr_df, out_path, scenario_name)
    
    print("3. Generating Reentry Analysis (Boxplots, KDE Maps, Prob Curves, & Density Lineplots)...")
    analyze_reentry_boxplots(df, metrics, out_path, scenario_name)
    analyze_reentry_kde(df, metrics, out_path, scenario_name)
    analyze_reentry_probability_curves(df, metrics, out_path, scenario_name)
    analyze_density_lineplots(df, structural_only_metrics, out_path, scenario_name)
    
    print("4. Generating Bivariate Pairplots...")
    generate_pairplots(df, structural_only_metrics, out_path, scenario_name)
    
    print(f"\nSUCCESS! All multivariate analyses saved to: {out_path}")

if __name__ == "__main__":
    main()