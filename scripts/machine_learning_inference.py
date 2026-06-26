"""
Machine Learning Inference Module for FibroStruct.

Uses XGBoost (Gradient Boosting) and SHAP (SHapley Additive exPlanations)
to evaluate the multivariate, non-linear impact of structural metrics on
the risk of sustained reentry.

Features:
1. Automated Hyperparameter Tuning via Stratified K-Fold GridSearchCV.
2. Handles Class Imbalance automatically.
3. Computes SHAP values for global and local interpretability.
4. Generates SHAP Summary (Beeswarm) and Dependence plots.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings

# Machine Learning & Explainability
import xgboost as xgb
import shap
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# --- GLOBAL PLOT SETTINGS ---
sns.set_theme(style="white", context="talk")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titleweight': 'bold',
    'figure.max_open_warning': 0,
})

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INVALID_NUMERIC = -9999.0

# Human-readable labels for plots
METRIC_LABELS = {
    'norm_mean_cluster_size': 'Norm. Mean Cluster Size',
    'mean_aspect_ratio': 'Mean Aspect Ratio',
    'fractal_dimension': 'Fractal Dimension',
    'morans_i': "Moran's I",
    'betti_0': 'Betti 0',
    'betti_1': 'Betti 1',
    'betti_2': 'Betti 2',
}

def clean_name(metric):
    return METRIC_LABELS.get(metric, metric.replace('_', ' ').title())

def get_scenario_name(dim, shape):
    mapping = {
        ('2D', 'full'): 'Scenario I',
        ('2D', 'ellipse'): 'Scenario II',
        ('3D', 'full'): 'Scenario III',
        ('3D', 'ellipse'): 'Scenario IV'
    }
    return mapping.get((dim, shape), f"Scenario ({dim}-{shape})")

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.replace(INVALID_NUMERIC, np.nan)
    
    df = df.dropna(axis=1, how='all')
    
    if 'sustained_reentry' not in df.columns:
        raise ValueError("Target column 'sustained_reentry' not found.")
        
    df['sustained_reentry'] = df['sustained_reentry'].fillna(0).astype(int)
    
    # Exclude non-structural variables and explicitly removed percolation metrics
    exclude = ['seed', 'actual_density', 'density', 'fiber_angle_deg', 'grid_x', 'grid_y', 'grid_z', 
               'dx', 'shape', 'atpi', 'homog_factor', 'final_time_ms', 'sustained_reentry', 
               'percolates_x', 'percolates_horiz', 'percolates_y', 'percolates_z', 
               'min_path_x', 'topological_conduction_eff', 'fibrosis_type']
               
    features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    
    # Drop rows where target or key features are completely missing
    df = df.dropna(subset=features + ['sustained_reentry'])
    return df, features

def optimize_and_train_xgboost(X, y):
    """
    Performs RandomizedSearchCV with StratifiedKFold to find robust hyperparameters,
    preventing overfitting while handling imbalanced classes.
    """
    # Calculate class weight for imbalance
    neg_class = sum(y == 0)
    pos_class = sum(y == 1)
    scale_pos = neg_class / pos_class if pos_class > 0 else 1.0

    # Base Model
    xgb_estimator = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=scale_pos,
        random_state=42
    )

    # Hyperparameter Grid
    param_distributions = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 500],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("   > Running RandomizedSearchCV (5-Fold CV) to find robust hyperparameters...")
    search = RandomizedSearchCV(
        estimator=xgb_estimator,
        param_distributions=param_distributions,
        n_iter=20, # Tests 20 random combinations
        scoring='roc_auc',
        cv=cv,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X, y)
    print(f"   > Best Hyperparameters found: {search.best_params_}")
    print(f"   > Best Cross-Validation AUC: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_score_

def generate_shap_analysis(model, X, features, out_dir, scenario_name):
    print("   > Calculating SHAP values...")
    
    # Rename columns to human-readable format for plots
    X_renamed = X.rename(columns={f: clean_name(f) for f in features})
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_renamed)
    
    # 1. SHAP Summary Plot (Beeswarm)
    # Shows feature importance AND the direction of the effect
    fig = plt.figure(figsize=(12, 8), dpi=300)
    shap.summary_plot(shap_values, X_renamed, show=False, plot_size=(12, 8))
    plt.title(f"{scenario_name} - SHAP Summary (Feature Impact on Reentry)", fontweight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(out_dir / "shap_summary_beeswarm.png", bbox_inches='tight')
    plt.close()

    # 2. SHAP Global Feature Importance (Bar Plot)
    fig = plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(shap_values, X_renamed, plot_type="bar", show=False, color="#21918c")
    plt.title(f"{scenario_name} - Global Feature Importance (Mean |SHAP|)", fontweight='bold', pad=20)
    plt.xlabel("Mean |SHAP Value| (Average impact on model output magnitude)")
    plt.tight_layout()
    fig.savefig(out_dir / "shap_importance_bar.png", bbox_inches='tight')
    plt.close()

    # 3. SHAP Dependence Plots (Top 3 Features)
    # Identifies non-linear thresholds where risk explodes
    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X_renamed.columns, vals)), columns=['col_name', 'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    
    top_3_features = feature_importance['col_name'].head(3).tolist()
    
    for feature in top_3_features:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        shap.dependence_plot(feature, shap_values, X_renamed, ax=ax, show=False)
        plt.title(f"SHAP Dependence: {feature}", fontweight='bold')
        plt.tight_layout()
        safe_name = feature.replace(' ', '_').replace('/', '_').replace('.', '').lower()
        fig.savefig(out_dir / f"shap_dependence_{safe_name}.png", bbox_inches='tight')
        plt.close()

def build_txt_report(model, cv_auc, X_test, y_test, features, out_dir, scenario_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    test_auc = roc_auc_score(y_test, y_prob)
    class_report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    lines = [
        "=========================================================================",
        f" MACHINE LEARNING (XGBOOST) INFERENCE REPORT - {scenario_name.upper()}",
        "=========================================================================\n",
        "--- MODEL ROBUSTNESS & VALIDATION ---",
        f"Cross-Validation AUC (Stratified 5-Fold) : {cv_auc:.4f}",
        f"Hold-Out Test Set AUC (Unseen Data)      : {test_auc:.4f}\n",
        "--- HYPERPARAMETERS SELECTED VIA GRID SEARCH ---"
    ]
    
    for key, val in model.get_params().items():
        if val is not None and key in ['max_depth', 'learning_rate', 'n_estimators', 'subsample', 'scale_pos_weight']:
            lines.append(f"{key.ljust(20)} : {val}")
            
    lines.extend([
        "\n--- CLASSIFICATION PERFORMANCE (TEST SET) ---",
        class_report,
        "\n--- CONFUSION MATRIX ---",
        f"True Negatives (Correctly predicted NO reentry) : {cm[0, 0]}",
        f"False Positives (Falsely predicted reentry)     : {cm[0, 1]}",
        f"False Negatives (Missed an actual reentry)      : {cm[1, 0]}",
        f"True Positives (Correctly predicted reentry)    : {cm[1, 1]}\n",
        "--- INTERPRETATION NOTE ---",
        "The SHAP Summary Plot (Beeswarm) generated alongside this report explains",
        "the 'HOW'. For any given feature, points pushed to the right (+ SHAP)",
        "indicate that the structural metric INCREASES the probability of reentry.",
        "The colors indicate the raw value of the feature (Red = High, Blue = Low)."
    ])
    
    with open(out_dir / "xgboost_inference_report.txt", "w") as f:
        f.write("\n".join(lines))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_dir", type=str, required=True, help="Base directory of the dataset.")
    parser.add_argument("--dim", type=str, required=True, choices=["2D", "3D"])
    parser.add_argument("--shape", type=str, required=True)
    
    args = parser.parse_args()
    scenario_name = get_scenario_name(args.dim, args.shape)
    
    scenario_batch_dir = PROJECT_ROOT / args.batch_dir / args.dim / args.shape
    scenario_master_csv = scenario_batch_dir / "data" / "master_dataset.csv"
    
    if not scenario_master_csv.exists():
        print(f"\n[!] FATAL ERROR: master_dataset.csv not found at {scenario_master_csv}")
        sys.exit(1)
        
    out_path = scenario_batch_dir / "analysis" / "machine_learning_shap"
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading and filtering dataset for {scenario_name}...")
    df, features = load_and_prepare_data(scenario_master_csv)
    
    print(f"Data ready. Total samples: {len(df)} | Target features: {len(features)}")
    
    # 80/20 Train-Test Split (Stratified to maintain proportion of arrhythmias)
    X = df[features]
    y = df['sustained_reentry']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("1. Optimizing XGBoost via Cross-Validation...")
    best_xgb_model, cv_auc = optimize_and_train_xgboost(X_train, y_train)
    
    print("2. Generating Inference & Performance Report...")
    build_txt_report(best_xgb_model, cv_auc, X_test, y_test, features, out_path, scenario_name)
    
    print("3. Executing SHAP Game Theory Analysis (Explainability)...")
    # For SHAP, we explain the patterns on the entire dataset to map the full physiological space
    generate_shap_analysis(best_xgb_model, X, features, out_path, scenario_name)
    
    print(f"\nSUCCESS! ML Inference and SHAP plots saved to: {out_path}")

if __name__ == "__main__":
    main()