import argparse
import pandas as pd
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description="Sorts the master_dataset.csv logically.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to master_dataset.csv")
    args = parser.parse_args()

    # Locate the master CSV
    csv_path = Path(args.csv_path)

    if not csv_path.exists():
        print(f"Error: master_dataset.csv not found in {args.csv_path}")
        sys.exit(1)

    print(f"Loading dataset from {csv_path.parent.name}...")
    df = pd.read_csv(csv_path)

    # Hierarchical sort order
    ideal_sort_cols = ['fibrosis_type', 'shape', 'grid_x', 'fiber_angle_deg', 'density', 'seed']
    actual_sort_cols = [c for c in ideal_sort_cols if c in df.columns]

    print(f"Sorting dataset hierarchically by: {', '.join(actual_sort_cols)}...")
    
    df = df.sort_values(by=actual_sort_cols).reset_index(drop=True)

    # Save it back in place
    df.to_csv(csv_path, index=False)
    print("Dataset successfully sorted and saved!")

if __name__ == "__main__":
    main()