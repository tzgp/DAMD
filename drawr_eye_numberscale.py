import matplotlib.pyplot as plt
import numpy as np

# Ensemble names
ensembles = ["1", "2", "3", "4", "5"]
I_AUROC = [0.877,0.903,0.904,0.904,0.904]
P_AUROC = [0.974, 0.98, 0.981, 0.981, 0.981]
AUPRO   = [0.875, 0.901, 0.901, 0.903, 0.904]
x = np.arange(len(ensembles))
plt.figure(figsize=(8,4))
plt.plot(x, I_AUROC, marker='o', linewidth=2, label='I-AUROC')
plt.plot(x, P_AUROC, marker='s', linewidth=2, label='P-AUROC')
plt.plot(x, AUPRO, marker='^', linewidth=2, label='AUPRO')
# Formatting
plt.xticks(x, ensembles, fontsize=12)
plt.ylim(0.85, 1.00)
plt.ylabel('Performance', fontsize=12)
#plt.xlabel(r'$\alpha$')
# plt.ylabel(r'$\beta$')
plt.xlabel(r'$\gamma$', fontsize=12)
plt.title('Eyecandies', fontsize=12)
#plt.title('Effect of the Number of Multi-Scale Levels on Eyecandies')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right')
# Optional: make it prettier
plt.tight_layout()
plt.savefig('gE.pdf', bbox_inches='tight')  # 矢量图版本
plt.savefig('ge.png', dpi=600, bbox_inches='tight')  # 矢量图版本
print("图像已保存为'Eyemulti_scale_levels_performance.pdf'")
plt.show()














