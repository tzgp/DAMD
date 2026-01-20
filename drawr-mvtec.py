# comparing different *multi-scale radius ensembles* (S1–S3 for min/mid/large ranges) across the three key metrics: **I-AUROC**, **P-AUROC**, and **AUPRO**.
#
# Below, I’ll give you:
#
# 1. 📊 A clear **Python/Matplotlib code** to generate a *Pattern Recognition–style figure* (ready to use).
# 2. 🧠 A short **caption and paragraph** to describe and interpret the figure in your paper.
#
# ---
#
# ## 📘 1. Python code for visualization
#
# Paste this into your plotting script (adjust metric values to your actual results from the table).
#
# ```python
import matplotlib.pyplot as plt
import numpy as np

# Ensemble names
#ensembles = ["S1 (Small-range-1)", "S2 (Small-range-2)", "S3 (Mid-range-1)", "S4 (Mid-range-2)", "S5 (Large-range-1)"]
ensembles = ["Small-range-1", "Small-range-2", "Mid-range-1", "Mid-range-2", "Large-range-1"]

I_AUROC = [0.936, 0.945, 0.954, 0.953, 0.956]
P_AUROC = [0.993, 0.993, 0.993, 0.993, 0.992]
AUPRO   = [0.968, 0.968, 0.969, 0.969, 0.966]

x = np.arange(len(ensembles))

plt.figure(figsize=(8,4))
plt.plot(x, I_AUROC, marker='o', linewidth=2, label='I-AUROC')
plt.plot(x, P_AUROC, marker='s', linewidth=2, label='P-AUROC')
plt.plot(x, AUPRO, marker='^', linewidth=2, label='AUPRO')
# Formatting
plt.xticks(x, ensembles, rotation=20, fontsize=12)
plt.ylim(0.92, 1.00)
plt.ylabel('Performance', fontsize=12)
plt.title('MVTec 3D-AD', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right')
# Optional: make it prettier
plt.xlabel(r'$\beta$', fontsize=12)
# Optional: make it prettier
plt.tight_layout()
plt.savefig('bM.pdf', bbox_inches='tight')  # 矢量图版本
print("图像已保存为'bM.pdf'")
plt.savefig('bM.png', dpi=600, bbox_inches='tight')
print("图像已保存为'bM.png'")
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
