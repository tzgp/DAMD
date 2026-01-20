import matplotlib.pyplot as plt
import numpy as np

# Ensemble names
ensembles = ["0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08", "0.09"]
I_AUROC = [0.92,	0.934,	0.945,	0.953,	0.954,	0.949,	0.95,	0.951,	0.95]
P_AUROC=[0.991,	0.992,	0.993,	0.993,	0.993,	0.993,	0.993,	0.993,	0.993]
AUPRO   =[0.963,	0.967,	0.969,	0.969,	0.969,	0.969,	0.969,	0.969,	0.969]



x = np.arange(len(ensembles))
plt.figure(figsize=(8,4))
plt.plot(x, I_AUROC, marker='o', linewidth=2, label='I-AUROC')
plt.plot(x, P_AUROC, marker='s', linewidth=2, label='P-AUROC')
plt.plot(x, AUPRO, marker='^', linewidth=2, label='AUPRO')
# Formatting
plt.xticks(x, ensembles, fontsize=12)
plt.ylim(0.85, 1.00)
plt.ylabel('Performance', fontsize=12)
#plt.title('Effect of voxel scale Settings on MVTec 3D-AD')
plt.title('MVTec 3D-AD')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right')
plt.xlabel(r'$\alpha$', fontsize=12)
# Optional: make it prettier
plt.tight_layout()
plt.savefig('aM.pdf', bbox_inches='tight')  # 矢量图版本
plt.savefig('aM.png', dpi=600, bbox_inches='tight')  # 矢量图版本
print("图像已保存为'aM.pdf'")
plt.show()







# ```
#
# ✅ *Tips for publication quality*
#
# * Use **sans-serif fonts** (set via `plt.rcParams['font.sans-serif'] = ['Arial']`).
# * For high DPI: `plt.savefig("parameter_effects.png", dpi=600, bbox_inches="tight")`.
# * Ensure consistent colors and font size with your other figures.
# * If submitting to Pattern Recognition, keep y-axis starting at 0.94 for visual emphasis but label it clearly.
#
# ---
#
# ## 📄 2. Example caption (LaTeX style)
#
# ```latex
# \begin{figure}[t]
# \centering
# \includegraphics[width=0.9\linewidth]{figs/parameter_effects.png}
# \caption{
# Comparison of different multi-scale radius ensembles on MVTec 3D-AD.
# The middle-range configuration S2 = \{0.03V, 0.07V, 0.11V, 0.15V\} achieves the best overall performance across all metrics (I-AUROC, P-AUROC, and AUPRO), confirming that moderate receptive fields provide an optimal balance between fine and global geometric context.
# }
# \label{fig:parameter_effects}
# \end{figure}
# ```
#
# ---
#
# ## 🧠 3. Example interpretation paragraph
#
# > As illustrated in Fig.~\ref{fig:parameter_effects}, the middle-range ensemble S2 = {0.03V, 0.07V, 0.11V, 0.15V} consistently achieves the highest scores across all three evaluation metrics (I-AUROC, P-AUROC, and AUPRO). Smaller receptive fields (S1) improve local sensitivity but fail to capture global structure, while overly large fields (S3) tend to smooth fine geometric variations. The balanced middle-range configuration effectively integrates local and global density cues, confirming its superior representational capability for multi-scale anomaly detection.
#
# ---
#
# Would you like me to generate the **plot image** (using your actual numeric values from the table you uploaded) so you can include it directly in your paper? If yes, I can extract the values and plot them for you.
