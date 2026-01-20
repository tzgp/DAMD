import matplotlib.pyplot as plt

# Example data
voxel_sizes = [0.03, 0.04, 0.05, 0.06, 0.07]
auc_voxel = [91.2, 93.5, 95.1, 94.7, 93.9]

density_scales = [1, 2, 3, 4, 5]
auc_density = [91.8, 93.4, 94.8, 95.3, 95.2]
pro_density = [89.6, 91.0, 92.7, 93.8, 93.6]

plt.figure(figsize=(10,4))

# Panel (a)
plt.subplot(1,2,1)
plt.plot(voxel_sizes, auc_voxel, '-o', label='AUC')
plt.axvline(x=0.05, color='r', linestyle='--', label='Optimal')
plt.xlabel('Base voxel size')
plt.ylabel('AUC (%)')
plt.title('(a) Influence of voxel size')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Panel (b)
plt.subplot(1,2,2)
plt.plot(density_scales, auc_density, '-o', label='AUC')
plt.plot(density_scales, pro_density, '--s', label='PRO')
plt.axvline(x=4, color='r', linestyle='--', label='Optimal')
plt.xlabel('Number of density radii')
plt.ylabel('Performance (%)')
plt.title('(b) Influence of density scales')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
