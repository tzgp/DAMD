import matplotlib.pyplot as plt
import numpy as np

# Ensemble names
ensembles = ["1", "2", "3", "4", "5"]
I_AUROC = [0.866,	0.92,	0.924	,0.954,	0.952]
P_AUROC = [0.986,	0.987,	0.988,	0.993,	0.993]
AUPRO   = [0.94,	0.946	,0.951	,0.969	,0.969]
x = np.arange(len(ensembles))
plt.figure(figsize=(8,4))
plt.plot(x, I_AUROC, marker='o', linewidth=2, label='I-AUROC')
plt.plot(x, P_AUROC, marker='s', linewidth=2, label='P-AUROC')
plt.plot(x, AUPRO, marker='^', linewidth=2, label='AUPRO')
# Formatting
plt.xticks(x, ensembles, fontsize=12)
plt.ylim(0.8, 1.00)
plt.ylabel('Performance', fontsize=12)
plt.xlabel(r'$\gamma$', fontsize=12)
#plt.title('Effect of the Number of Multi-Scale Levels on MVTec 3D-AD')
plt.title('MVTec 3D-AD', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right')
# Optional: make it prettier
plt.tight_layout()
plt.savefig('gM.pdf', bbox_inches='tight')  # 矢量图版本
plt.savefig('gM.png', dpi=600, bbox_inches='tight')  # 矢量图版本
print("图像已保存为'multi_scale_levels_performance.pdf'")
plt.show()














