#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Arguments passed from the command line
BATCH_DIR=$1
DIM=$2
SHAPE=$3

# Check if both arguments are provided
if [ -z "$BATCH_DIR" ] || [ -z "$DIM" ] || [ -z "$SHAPE" ]; then
    echo "Error: Missing arguments."
    echo "Usage: ./run_pipeline.sh <path_to_batch_dir> <dim> <shape>"
    echo "Example: ./run_pipeline.sh ../simulations_batch_dir 2D full"
    exit 1
fi

echo "====================================================================================="
echo " Initializing Comprehensive Analysis Pipeline for: $BATCH_DIR"
echo "====================================================================================="

# Directories and file paths
SCENARIO_DIR="$BATCH_DIR/$DIM/$SHAPE"
DATA_DIR="$SCENARIO_DIR/data"
MASTER_CSV="$DATA_DIR/master_dataset.csv"

if [ ! -f "$MASTER_CSV" ]; then
    echo "CRITICAL ERROR: master_dataset.csv not found in $DATA_DIR."
    echo "Please ensure orchestrator.py completed successfully."
    exit 1
fi

# 1. Execute Dataset Sorting and Wrangling (sort_dataset.py)
echo "[1/5] Sorting Master Dataset..."
python3 scripts/organize_dataset.py --csv_path "$MASTER_CSV"

# 2. Execute Structural Analysis (analyze_structure.py)
echo -e "\n[2/5] Running Structural & Topological Analysis..."
python3 scripts/analyze_structure.py --batch_dir "$BATCH_DIR" --dim "$DIM" --shape "$SHAPE"

# 3. Execute Functional Analysis & Pre-processing (analyze_function.py)
echo -e "\n[3/5] Running Empirical Analysis (EP) & GAM Data Prep..."
python3 scripts/analyze_empirical.py --batch_dir "$BATCH_DIR" --dim "$DIM" --shape "$SHAPE"

# 4. Execute Statistical Modeling (statistical_models.R)
# It will look for data_for_gam_r.csv inside the empirical output folder
echo -e "\n[4/5] Running R Statistics (Firth and GAM Models)..."
Rscript scripts/statistical_models.R "$BATCH_DIR" "$DIM" "$SHAPE"

# 5. Execute Multivariate Analysis (multivariate_analysis.py)
echo -e "\n[5/5] Running Multivariate Analysis..."
python3 scripts/multivariate_analysis.py --batch_dir "$BATCH_DIR" --dim "$DIM" --shape "$SHAPE"

echo "====================================================================================="
echo " Pipeline Completed Successfully!"
echo " All results, plots, and reports are saved within: $SCENARIO_DIR/analysis"
echo "====================================================================================="