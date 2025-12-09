import pandas as pd
import argparse
import sys

def calculate_means():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description='Calculate mean metrics for Autoencoder results.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file containing results')
    
    args = parser.parse_args()

    # 2. Load the Data
    try:
        df = pd.read_csv(args.file_path)
        print(f"Successfully loaded: {args.file_path} ({len(df)} rows)")
    except FileNotFoundError:
        print(f"Error: The file '{args.file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # 3. Define Columns to Look For
    # We look for these exact names, but you can add aliases if needed
    metric_cols = ['chamfer_distance', 'hausdorff_distance', 'p2s_dist', 'compression_time_sec', 'decompression_time_sec']
    
    # Potential names for grouping columns (Model Name and Latent Dimension)
    model_col_options = ['Model', 'model', 'architecture', 'variant']
    dim_col_options = ['Dim', 'dim', 'dimension', 'latent_dim', 'd']

    # 4. Find which columns actually exist in the CSV
    found_metrics = [c for c in metric_cols if c in df.columns]
    
    found_model_col = next((c for c in model_col_options if c in df.columns), None)
    found_dim_col = next((c for c in dim_col_options if c in df.columns), None)

    if not found_metrics:
        print("Error: Could not find any metric columns (CD, HD, P2S, T_comp, T_decomp).")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # 5. Group and Calculate Means
    grouping = []
    if found_model_col: grouping.append(found_model_col)
    if found_dim_col: grouping.append(found_dim_col)

    if grouping:
        print(f"Grouping by: {grouping}")
        # Calculate mean
        result = df.groupby(grouping)[found_metrics].mean().reset_index()
        
        # Sort by Dimension if it exists, to make the table look nice
        if found_dim_col:
            result = result.sort_values(by=grouping)
            
        # Round values for cleaner reading
        print("\n--- Averaged Results ---")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(result.to_string(index=False))
        
        # Bonus: Print a LaTeX-friendly snippet
        print("\n--- LaTeX Body Snippet (Quick Copy) ---")
        for index, row in result.iterrows():
            # Format: Model & Dim & CD & HD & P2S & T_comp & T_decomp \\
            m_name = row[found_model_col] if found_model_col else "Model"
            d_val = int(row[found_dim_col]) if found_dim_col else 0
            
            # Format strings: {:.4f} for errors, {:.3f} for time
            line = f"{m_name} & {d_val} & "
            line += " & ".join([f"{row[m]:.4f}" for m in found_metrics])
            line += " \\\\"
            print(line)

    else:
        print("\nNo grouping columns found (Model/Dim). Calculating global average:")
        print(df[found_metrics].mean())

if __name__ == "__main__":
    calculate_means()