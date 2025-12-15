import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# ==========================================
# CONFIGURATION: SET YOUR PATHS HERE
# ==========================================
# Paths to the single Draco CSV files
DRACO_SHREC16_FILE = 'src/WrappingNet/results/draco_benchmark_shrec_16.csv'
AE_SHREC16_FOLDER = 'src/WrappingNet/results/ae_basic_shrec16_test_cr/'

# Output filename
OUTPUT_IMAGE = 'inverse_correlation.png'

# ==========================================
# DATA LOADING FUNCTIONS
# ==========================================

def load_draco(file_path, dataset_name):
    """Loads a single Draco benchmark CSV."""
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    # Aggregate by q_level to get one point per setting
    agg = df.groupby('q_level').mean(numeric_only=True).reset_index()
    
    output_rows = []
    for _, row in agg.iterrows():
        output_rows.append({
            'Method': 'Draco',
            'Dataset': dataset_name,
            'Label': f"q{int(row['q_level'])}",
            'BPV': row['bpv'],
            'Compression_Ratio': row['compression_ratio'],
            'Is_Projected': False
        })
    return pd.DataFrame(output_rows)

def load_ae_folder(folder_path, dataset_name):
    """Scans a folder for AE CSV files (looking for 'd32', 'd64' patterns)."""
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}")
        return pd.DataFrame()

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    output_rows = []
    
    for f in csv_files:
        filename = os.path.basename(f)
        
        # Robustly extract dimension (d32, d64, etc.) from filename
        # Assumes filename contains pattern like '_d32_' or 'd32'
        try:
            # Simple heuristic: Split by underscores, find part starting with 'd' + numbers
            parts = filename.split('_')
            dim_str = next((s for s in parts if s.startswith('d') and s[1:].isdigit()), None)
            
            if dim_str:
                dim = int(dim_str[1:])
                df = pd.read_csv(f)
                
                # Check column names (handle variation in AE csvs)
                ratio_col = 'ratio_latent_only' if 'ratio_latent_only' in df.columns else 'compression_ratio'
                
                output_rows.append({
                    'Method': 'AE',
                    'Dataset': dataset_name,
                    'Label': dim_str,
                    'BPV': df['bpv'].mean(),
                    'Compression_Ratio': df[ratio_col].mean(),
                    'Is_Projected': False
                })
        except Exception as e:
            print(f"Skipping file {filename}: {e}")

    return pd.DataFrame(output_rows)

# ==========================================
# MAIN EXECUTION
# ==========================================

# 1. Load Data
print("Loading Data...")
df_draco_shrec = load_draco(DRACO_SHREC16_FILE, 'SHREC16')
df_ae_shrec = load_ae_folder(AE_SHREC16_FOLDER, 'SHREC16')

# Combine all "Real" data
df = pd.concat([df_draco_shrec, df_ae_shrec], ignore_index=True)

if df.empty:
    raise ValueError("No data loaded! Check your paths.")

# 2. Calculate Projection Multiplier
# We compare AE at common points (d32, d64, d128)
ae_shrec_idx = df[(df['Method']=='AE') & (df['Dataset']=='SHREC16')].set_index('Label')

multipliers = []

# Default to 2.0 if no overlap found, otherwise use calculated average
avg_multiplier = np.mean(multipliers) if multipliers else 2.0
print(f"Calculated Multiplier: {avg_multiplier:.4f}")

projected_rows = []
dims_to_project = ['d8', 'd16', 'd32', 'd64', 'd128', 'd256' 'd512']

df_projected = pd.DataFrame(projected_rows)
df_final = pd.concat([df, df_projected], ignore_index=True)

# ==========================================
# PLOTTING
# ==========================================
print("Plotting...")
plt.figure(figsize=(12, 8))

# Style Settings
markers = {'SHREC16': 'o'}
colors = {'Draco': 'tab:red', 'AE': 'tab:blue'}
linestyles = {'SHREC16': '--'}

# --- NEW: CALCULATE REGION BOUNDARIES ---
# Find min BPV for Draco (The cutoff point)
draco_data = df_final[(df_final['Method'] == 'Draco') & (df_final['Dataset'] == 'SHREC16')]
ae_data = df_final[(df_final['Method'] == 'AE') & (df_final['Dataset'] == 'SHREC16')]

if not draco_data.empty and not ae_data.empty:
    min_draco_bpv = draco_data['BPV'].min()
    min_ae_bpv = ae_data['BPV'].min()

    # Only draw if AE actually goes lower than Draco
    if min_ae_bpv < min_draco_bpv:
        # 1. Add Shaded Region
        plt.axvspan(min_ae_bpv * 0.9, min_draco_bpv, 
                    color='tab:blue', alpha=0.1, lw=0) # Light blue shading
        
        # 2. Add Annotation Text
        # Place text in the middle of the region, near the top
        mid_point = (min_ae_bpv + min_draco_bpv) / 2
        # Use a geometric mean for log-scale visual centering
        log_mid_point = 10**((np.log10(min_ae_bpv) + np.log10(min_draco_bpv))/2)
        
        plt.text(log_mid_point, ae_data['Compression_Ratio'].max() * 0.5, 
                 "AE Exclusive Region\n(Ultra-Low Bitrate)", 
                 color='tab:blue', fontsize=11, fontweight='bold', 
                 ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
        
        # Optional: Add a vertical line at the boundary
        plt.axvline(x=min_draco_bpv, color='tab:red', linestyle=':', alpha=0.5)
        plt.text(min_draco_bpv, ae_data['Compression_Ratio'].min(), " Draco Limit ", 
                 color='tab:red', rotation=90, va='bottom', ha='right', fontsize=9)

# Iterate through groups to plot lines
for method in ['Draco', 'AE']:
    for dataset in ['SHREC16']:
        # Get data for this line
        subset = df_final[(df_final['Method'] == method) & (df_final['Dataset'] == dataset)].sort_values('BPV')
        
        if subset.empty:
            continue
            
        # Separate real vs projected for styling
        real_subset = subset[subset['Is_Projected'] == False]
        proj_subset = subset[subset['Is_Projected'] == True]
        
        # 1. Plot Real Line
        if not real_subset.empty:
            plt.plot(real_subset['BPV'], real_subset['Compression_Ratio'], 
                     marker=markers[dataset], color=colors[method], linestyle=linestyles[dataset], 
                     linewidth=2, label=f"{method}")
            
            # Annotate endpoints
            start = real_subset.iloc[0]
            end = real_subset.iloc[-1]
            plt.annotate(start['Label'], (start['BPV'], start['Compression_Ratio']), xytext=(-10, 10), textcoords='offset points', color=colors[method], fontsize=9)
            plt.annotate(end['Label'], (end['BPV'], end['Compression_Ratio']), xytext=(5, 5), textcoords='offset points', color=colors[method], fontsize=9)

        # 2. Plot Projected Line (Dotted)
        if not proj_subset.empty:
            # Connect the gap (Last Real -> First Projected)
            if not real_subset.empty:
                gap_x = [real_subset.iloc[-1]['BPV'], proj_subset.iloc[0]['BPV']]
                gap_y = [real_subset.iloc[-1]['Compression_Ratio'], proj_subset.iloc[0]['Compression_Ratio']]
                plt.plot(gap_x, gap_y, color=colors[method], alpha=0.7)
            
            # Plot the projected segment
            plt.plot(proj_subset['BPV'], proj_subset['Compression_Ratio'], 
                     marker=markers[dataset], color=colors[method], alpha=0.6, markersize=6)
            
            # Annotate projected points
            for _, row in proj_subset.iterrows():
                plt.annotate(f"{row['Label']}*", (row['BPV'], row['Compression_Ratio']), 
                             xytext=(5, 5), textcoords='offset points', color=colors[method], alpha=0.7, fontsize=8)

# Formatting
plt.title('Inverse Correlation: Compression Ratio vs BPV\n', fontsize=14)
plt.xlabel('Bits Per Vertex (BPV)', fontsize=12)
plt.ylabel('Compression Ratio', fontsize=12)
plt.yscale('log')
plt.xscale('log')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(fontsize=10)

# Save
plt.savefig(OUTPUT_IMAGE, dpi=300)
print(f"Graph saved to {OUTPUT_IMAGE}")
plt.show()