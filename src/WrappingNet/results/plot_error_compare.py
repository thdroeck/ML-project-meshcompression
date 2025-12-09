import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import sys

def load_folder_data(folder_path, label):
    """Loads all CSVs in a folder and adds a 'Model' identifier column."""
    search_path = os.path.join(folder_path, "*.csv")
    files = glob.glob(search_path)
    
    if not files:
        print(f"Warning: No CSV files found in {folder_path}. Skipping.")
        return None

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    combined['Model'] = label  # Tag this data with the model name
    return combined

def main():
    # 1. Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Compare reconstruction errors across multiple folders.")
    
    # Allow multiple folders: python plot.py --folders path/to/A path/to/B --labels Simple Extended
    parser.add_argument("--folders", nargs='+', required=True, help="List of folder paths to compare.")
    parser.add_argument("--labels", nargs='+', help="List of names for the legend (must match count of folders).")
    parser.add_argument("--output", type=str, default="comparison_metrics.png", help="Output filename for the graph.")
    
    args = parser.parse_args()

    # Validate labels
    if args.labels:
        if len(args.labels) != len(args.folders):
            print("Error: The number of labels must match the number of folders.")
            sys.exit(1)
        labels = args.labels
    else:
        # If no labels provided, use the folder names
        labels = [os.path.basename(os.path.normpath(f)) for f in args.folders]

    # 2. Load and combine data from all folders
    all_data = []
    print(f"Processing {len(args.folders)} folders...")

    for folder, label in zip(args.folders, labels):
        df = load_folder_data(folder, label)
        if df is not None:
            all_data.append(df)
        else:
            print(f"Skipping empty or invalid folder: {folder}")

    if not all_data:
        print("No valid data found in any folder.")
        sys.exit(1)

    full_df = pd.concat(all_data, ignore_index=True)

    # 3. Prepare data for plotting
    required_columns = ['latent_dim', 'chamfer_distance', 'hausdorff_distance', 'p2s_dist', 'Model']
    
    # Check for missing columns (except Model which we just added)
    missing_cols = [c for c in required_columns[:-1] if c not in full_df.columns]
    if missing_cols:
        print(f"Error: Dataframes are missing required columns: {missing_cols}")
        sys.exit(1)

    # Melt to long format: 
    # id_vars keeps 'latent_dim' AND 'Model'
    melted_df = full_df.melt(
        id_vars=['latent_dim', 'Model'], 
        value_vars=['chamfer_distance', 'hausdorff_distance', 'p2s_dist'],
        var_name='Metric', 
        value_name='Error'
    )

    # Rename metrics for cleaner titles if desired (optional)
    metric_map = {
        'chamfer_distance': 'Chamfer Distance',
        'hausdorff_distance': 'Hausdorff Distance',
        'p2s_dist': 'P2S Distance'
    }
    melted_df['Metric'] = melted_df['Metric'].map(metric_map)

    # 4. Generate the graph (Side-by-Side Comparison)
    sns.set_theme(style="whitegrid")

    # relational plot with 'col' used to create subplots for each metric
    g = sns.relplot(
        data=melted_df,
        x="latent_dim", 
        y="Error",
        hue="Model",    # Different colors for different folders
        col="Metric",   # Separate plot for each metric
        kind="line",
        marker="o",
        dashes=False,   # Solid lines for all
        facet_kws={'sharey': False, 'sharex': True}, # Let Y-axis scale independently per metric
        height=5, 
        aspect=1
    )

    # Add titles and labels
    g.set_titles("{col_name}")  # Set subplot titles to metric names
    g.set_axis_labels("Latent Dimension", "Error Value")
    
    # Adjust layout title
    g.figure.subplots_adjust(top=0.85)
    g.figure.suptitle('Model Comparison: Reconstruction Error vs Latent Dimension', fontsize=16)

    # 5. Save
    g.savefig(args.output)
    print(f"Comparison graph saved to {args.output}")

if __name__ == "__main__":
    main()