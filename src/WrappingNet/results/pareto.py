import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# ==========================================
# 1. SETUP PATHS
# ==========================================
DRACO_FILE = 'src/WrappingNet/results/draco_benchmark_shrec_16.csv'
AE_FOLDER = 'src/WrappingNet/results/ae_basic_shrec16_test_cr/' 

# ==========================================
# 2. LOAD & PROCESS DATA
# ==========================================

# Load Draco
draco_df = pd.read_csv(DRACO_FILE)
draco_agg = draco_df.groupby('q_level').mean(numeric_only=True).sort_values('bpv')

# Load AE Basic
ae_data = []
# Find all matching AE Basic files
file_pattern = os.path.join(AE_FOLDER, "ae_basic_shrec16_test_cr_d*_e10.csv")
for filepath in glob.glob(file_pattern):
    filename = os.path.basename(filepath)
    try:
        # Extract dimension (d32, d64, etc.)
        parts = filename.split('_')
        dim_str = next(p for p in parts if p.startswith('d') and p[1:].isdigit())
        dim = int(dim_str[1:])
        
        df = pd.read_csv(filepath)
        ae_data.append({
            'Dimension': dim,
            'Label': f"d{dim}",
            'BPV': df['bpv'].mean(),
            'Error': df['p2s_dist'].mean()
        })
    except Exception as e:
        print(f"Skipping {filename}: {e}")

ae_agg = pd.DataFrame(ae_data).sort_values('BPV')

# ==========================================
# 3. CALCULATE PARETO FRONTIER
# ==========================================
# Combine all points into one list
all_points = []
for idx, row in draco_agg.iterrows():
    all_points.append({'x': row['bpv'], 'y': row['p2s_dist'], 'type': 'Draco'})
for _, row in ae_agg.iterrows():
    all_points.append({'x': row['BPV'], 'y': row['Error'], 'type': 'AE'})

# Sort by BPV (x-axis)
all_points.sort(key=lambda k: k['x'])

# Find the frontier: strict monotonic decrease in Error
frontier_points = []
min_error_so_far = float('inf')

for p in all_points:
    if p['y'] < min_error_so_far:
        frontier_points.append(p)
        min_error_so_far = p['y']

frontier_df = pd.DataFrame(frontier_points)

# ==========================================
# 4. PLOTTING
# ==========================================
plt.figure(figsize=(10, 7))

# Plot the "Raw" Curves
plt.plot(draco_agg['bpv'], draco_agg['p2s_dist'], 
         marker='o', linestyle='-', color='tab:red', alpha=0.4, label='Draco (Raw)')
plt.plot(ae_agg['BPV'], ae_agg['Error'], 
         marker='s', linestyle='--', color='tab:blue', alpha=0.4, label='AE Basic (Raw)')

# Plot the Pareto Frontier (Highlight)
plt.plot(frontier_df['x'], frontier_df['y'], 
         color='purple', linewidth=3, linestyle='--', label='Pareto Frontier (Optimal)')
plt.scatter(frontier_df['x'], frontier_df['y'], color='purple', s=40, zorder=5)

# Annotations
for _, row in ae_agg.iterrows():
    plt.annotate(row['Label'], (row['BPV'], row['Error']), 
                 xytext=(0, -15), textcoords='offset points', ha='center', fontsize=9, color='tab:blue')
    
# Annotate Draco Points (q2, q4, ...)
for i, row in draco_agg.iterrows():
    # Annotate only selected points to avoid clutter
    if int(i) in [2, 4, 6, 8, 10, 12, 14]: 
        plt.annotate(f"q{int(i)}", (row['bpv'], row['p2s_dist']), 
                     xytext=(0, 10), textcoords='offset points', ha='center', color='tab:red', fontsize=9)

plt.title('Pareto Efficiency Frontier: Basic AE vs. Draco', fontsize=14)
plt.xlabel('Bits Per Vertex (BPV)', fontsize=12)
plt.ylabel('Reconstruction Error (P2S)', fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()

plt.savefig('pareto_frontier_plot.png', dpi=300)
plt.show()