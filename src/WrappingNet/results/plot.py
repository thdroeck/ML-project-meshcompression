import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the datasets
df_e1 = pd.read_csv('src/WrappingNet/results/autoencoder_benchmark_test_airplane_d64_e10.csv')
df_e10 = pd.read_csv('src/WrappingNet/results/draco_benchmark.csv')

# 2. Prepare the data for plotting
df_e1['Epoch'] = 'E10'
df_e10['Epoch'] = 'Draco'
df_combined = pd.concat([df_e1, df_e10], axis=0)

# ==========================================
# PLOT 1: Comparison Box Plots (4 subplots)
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Chamfer Distance
sns.boxplot(x='Epoch', y='chamfer_distance', data=df_combined, ax=axes[0, 0])
axes[0, 0].set_title('Chamfer Distance Distribution (Lower is Better)')

# Hausdorff Distance
sns.boxplot(x='Epoch', y='hausdorff_distance', data=df_combined, ax=axes[0, 1])
axes[0, 1].set_title('Hausdorff Distance Distribution (Lower is Better)')

# Compression Time
sns.boxplot(x='Epoch', y='compression_time_sec', data=df_combined, ax=axes[1, 0])
axes[1, 0].set_title('Compression Time Distribution')

# Decompression Time
sns.boxplot(x='Epoch', y='decompression_time_sec', data=df_combined, ax=axes[1, 1])
axes[1, 1].set_title('Decompression Time Distribution')

plt.tight_layout()
plt.savefig('src/WrappingNet/results/comparison_plots.png')
plt.show()

# ==========================================
# PLOT 2: Improvement Scatter Plot
# ==========================================
plt.figure(figsize=(6, 6))

# Scatter plot: x-axis is Epoch 1 error, y-axis is Epoch 10 error
plt.scatter(df_e1['chamfer_distance'], df_e10['chamfer_distance'], alpha=0.7)

# Add a diagonal line. Points below this line indicate improvement.
max_val = df_e1['chamfer_distance'].max()
plt.plot([0, max_val], [0, max_val], 'r--', label='No Improvement')

plt.xlabel('Chamfer Distance (E10)')
plt.ylabel('Chamfer Distance (Draco)')
plt.title('Chamfer Distance: E10 vs Draco per sample')
plt.legend()
plt.grid(True)

plt.savefig('src/WrappingNet/results/chamfer_improvement_scatter.png')
plt.show()