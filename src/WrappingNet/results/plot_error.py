import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import sys

def main():
    # 1. Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Generate reconstruction error graph from CSV files.")
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing CSV files.")
    parser.add_argument("--output", type=str, default="combined_metrics.png", help="Output filename for the graph.")
    
    args = parser.parse_args()
    
    # 2. Find all CSV files in the specified folder
    # This creates a path like: ./ae_basic_manifold_test/*.csv
    search_path = os.path.join(args.folder, "*.csv")
    files = glob.glob(search_path)
    
    if not files:
        print(f"No CSV files found in {args.folder}")
        sys.exit(1)
        
    print(f"Found {len(files)} files. Processing...")

    # 3. Load and combine data
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not dfs:
        print("No valid data found.")
        sys.exit(1)

    full_df = pd.concat(dfs, ignore_index=True)

    # 4. Prepare data for plotting
    # We require these columns to exist
    required_columns = ['latent_dim', 'chamfer_distance', 'hausdorff_distance', 'p2s_dist']
    if not all(col in full_df.columns for col in required_columns):
        print(f"Error: Dataframe missing one of the required columns: {required_columns}")
        print(f"Available columns: {full_df.columns.tolist()}")
        sys.exit(1)

    # Melt the dataframe to 'long' format for Seaborn
    metrics_df = full_df[required_columns]
    melted_df = metrics_df.melt(id_vars=['latent_dim'], 
                                value_vars=['chamfer_distance', 'hausdorff_distance', 'p2s_dist'],
                                var_name='Metric', value_name='Error')

    # 5. Generate the graph
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    
    # Using 'style' and 'hue' to differentiate lines
    sns.lineplot(data=melted_df, x='latent_dim', y='Error', hue='Metric', style='Metric', marker='o', errorbar='sd')

    plt.title('Reconstruction Error Metrics vs. Latent Dimension')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Error Value')
    plt.legend(title='Metric')
    
    # 6. Save the plot
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Graph saved to {args.output}")

if __name__ == "__main__":
    main()