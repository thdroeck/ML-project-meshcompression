import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Replace these paths with the actual folders where your CSVs are stored
# The script expects files like "ae_basic_shrec16_test_cr_d8_e10.csv" inside these folders
FOLDERS = {
    'Basic':    'src/WrappingNet/results/ae_basic_shrec16_test_cr/',
    'Simple':   'src/WrappingNet/results/ae_simple_shrec16_test_cr/',
    'Extended': 'src/WrappingNet/results/ae_extended_shrec16_test_cr/'
}

# Visualization Settings
STYLES = {
    'Basic':    {'color': 'tab:blue',  'marker': 'v', 'label': 'Basic'},
    'Simple':   {'color': 'tab:green',    'marker': 'v', 'label': 'Simple'},
    'Extended': {'color': 'tab:orange', 'marker': 'v', 'label': 'Extended'}
}

# ==========================================
# 2. DATA LOADING
# ==========================================
data_points = []

print("Loading data...")

for model_name, folder_path in FOLDERS.items():
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}")
        continue

    # Look for all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        
        # Robustly extract dimension (d8, d16, d32...)
        # Assumes filename contains "_d{number}_" pattern
        try:
            parts = filename.split('_')
            dim_str = next((s for s in parts if s.startswith('d') and s[1:].isdigit()), None)
            
            if dim_str:
                dim = int(dim_str[1:])
                df = pd.read_csv(filepath)
                
                # Check for correct column names
                ratio_col = 'ratio_latent_only' if 'ratio_latent_only' in df.columns else 'compression_ratio'
                error_col = 'p2s_dist'
                
                if ratio_col in df.columns and error_col in df.columns:
                    data_points.append({
                        'Model': model_name,
                        'Dimension': dim,
                        'Label': f"d{dim}",
                        'Ratio': df[ratio_col].mean(),
                        'Error': df[error_col].mean()
                    })
        except Exception as e:
            print(f"Skipping {filename}: {e}")

df = pd.DataFrame(data_points)

if df.empty:
    raise ValueError("No data found! Check your folder paths.")

# ==========================================
# 3. PLOTTING
# ==========================================
plt.figure(figsize=(12, 8))

# Plot lines for each model
for model_name, style in STYLES.items():
    subset = df[df['Model'] == model_name].sort_values('Dimension')
    
    if not subset.empty:
        plt.plot(subset['Error'], subset['Ratio'], 
                 color=style['color'], 
                 marker=style['marker'], 
                 linestyle='', 
                 linewidth=2, 
                 markersize=10, 
                 label=style['label'])
        
        # Annotate each point with its dimension (d8, d32...)
        for _, row in subset.iterrows():
            plt.annotate(row['Label'], 
                         (row['Error'], row['Ratio']), 
                         xytext=(5, 5), textcoords='offset points', 
                         color=style['color'], fontweight='bold', fontsize=9)

# Add "Negative Compression" Zone
plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=2)
plt.fill_between([df['Error'].min()*0.9, df['Error'].max()*1.1], 0, 1.0, color='red', alpha=0.05)
plt.text(df['Error'].min(), 0.8, 'NEGATIVE COMPRESSION ZONE (Ratio < 1.0)', 
         color='red', fontsize=10, fontweight='bold', va='top')

# Formatting
plt.title('Compression Ratio vs Reconstruction Error', fontsize=14)
plt.xlabel('Reconstruction Error (P2S) - Lower is Better', fontsize=12)
plt.ylabel('Compression Ratio (Log Scale) - Higher is Better', fontsize=12)
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(fontsize=12, loc='upper right')

# Output
output_file = 'ae_variants_comparison.png'
plt.savefig(output_file, dpi=300)
print(f"Graph saved to {output_file}")
plt.show()