import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# ==========================================
# 1. SETUP PATHS
# ==========================================
DRACO_FILE = 'src/WrappingNet/results/draco_benchmark_shrec_16.csv'
AE_FOLDER = 'src/WrappingNet/results/ae_basic_shrec16_test_cr/' 

# ==========================================
# 2. LOAD DRACO DATA
# ==========================================
draco_df = pd.read_csv(DRACO_FILE)
# Group by q_level to get average performance per setting
draco_agg = draco_df.groupby('q_level').mean(numeric_only=True).sort_values('bpv')

# ==========================================
# 3. LOAD AE BASIC DATA
# ==========================================
ae_data = []
# Pattern matches all ae_basic_shrec16... CSVs
file_pattern = os.path.join(AE_FOLDER, "ae_basic_shrec16_test_cr_d*_e10.csv")

for filepath in glob.glob(file_pattern):
    # Extract dimension from filename (e.g., ..._d32_...)
    filename = os.path.basename(filepath)
    try:
        # Split by '_' and find the part starting with 'd'
        parts = filename.split('_')
        dim_str = next(p for p in parts if p.startswith('d') and p[1:].isdigit())
        dim = int(dim_str[1:])
        
        df = pd.read_csv(filepath)
        
        ae_data.append({
            'Dimension': dim,
            'Label': f"d{dim}",
            'BPV': df['bpv'].mean(),
            'Error': df['p2s_dist'].mean() # Or 'chamfer_distance' if preferred
        })
    except Exception as e:
        print(f"Skipping {filename}: {e}")

ae_agg = pd.DataFrame(ae_data).sort_values('BPV')

# ==========================================
# 4. PLOT THE RD-CURVE
# ==========================================
plt.figure(figsize=(10, 7))

# --- Plot Draco ---
plt.plot(draco_agg['bpv'], draco_agg['p2s_dist'], 
         marker='o', linestyle='-', linewidth=2, color='tab:red', label='Google Draco')

# Annotate Draco Points (q2, q4, ...)
for i, row in draco_agg.iterrows():
    # Annotate only selected points to avoid clutter
    if int(i) in [2, 4, 6, 8, 10, 12, 14]: 
        plt.annotate(f"q{int(i)}", (row['bpv'], row['p2s_dist']), 
                     xytext=(0, 10), textcoords='offset points', ha='center', color='tab:red', fontsize=9)

# --- Plot AE Basic ---
plt.plot(ae_agg['BPV'], ae_agg['Error'], 
         marker='s', linestyle='--', linewidth=2, color='tab:blue', label='Basic AE')

# Annotate AE Points (d8, d32, ...)
for _, row in ae_agg.iterrows():
    plt.annotate(row['Label'], (row['BPV'], row['Error']), 
                 xytext=(0, -15), textcoords='offset points', ha='center', color='tab:blue', fontsize=9, fontweight='bold')

# --- Formatting ---
plt.title('Rate-Distortion Curve: AE Basic vs. Draco', fontsize=14)
plt.xlabel('Bits Per Vertex (BPV) - Lower is Better Compression', fontsize=12)
plt.ylabel('Reconstruction Error (P2S) - Lower is Better Quality', fontsize=12)

# Use Log Scale? 
# Usually YES for Error, sometimes for BPV depending on range
plt.yscale('log') 
plt.xscale('log') 

plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(fontsize=12)

# Save
plt.savefig('rd_curve_comparison.png', dpi=300)
plt.show()