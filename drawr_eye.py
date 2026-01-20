import matplotlib.pyplot as plt
import numpy as np

# Ensemble names
ensembles = ["Small-range-1", "Small-range-2", "Mid-range-1", "Mid-range-2", "Large-range-1"]

I_AUROC = [0.897, 0.894, 0.898, 0.904, 0.907]
P_AUROC = [0.98, 0.98, 0.979, 0.981, 0.98]
AUPRO   = [0.903, 0.9, 0.896, 0.903, 0.899]
x = np.arange(len(ensembles))
plt.figure(figsize=(8,4))
plt.plot(x, I_AUROC, marker='o', linewidth=2, label='I-AUROC')
plt.plot(x, P_AUROC, marker='s', linewidth=2, label='P-AUROC')
plt.plot(x, AUPRO, marker='^', linewidth=2, label='AUPRO')
# Formatting
plt.xticks(x, ensembles, rotation=20, fontsize=12)
plt.ylim(0.85, 1.00)
plt.ylabel('Performance', fontsize=12)
plt.title('Eyecandies', fontsize=12)
#plt.title('Effect of Multi-scale Radius Settings on Eyecandies')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right')
# Optional: make it prettier
plt.xlabel(r'$\beta$', fontsize=12)
# Optional: make it prettier
plt.tight_layout()
plt.savefig('bE.pdf', bbox_inches='tight')  # 矢量图版本
print("图像已保存为'bE.pdf'")
plt.savefig('bE.png', dpi=600, bbox_inches='tight')
print("图像已保存为'bE.png'")
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
