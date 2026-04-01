# OBS - outputted/created figures are only placeholders for now. Data is currently randomized/pseudo. 
# Will switch to our actual data from SVM and BINN later!

import matplotlib
matplotlib.use('Agg') # Backend (non-interactive for linux servers):

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Configuration (global constants and paths):
CELL_TYPES = ['astro', 'exc1', 'exc2', 'exc3', 'immune', 'inhi', 'oligo', 'opcs', 'vasc']
SAVE_DIR = "figures/"
N_BOOTSTRAP = 1000
RANDOM_SEED = 42

# Initialization (environment setup):
os.makedirs(SAVE_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)

# Data Loading (svm placeholder scores):
def load_svm_scores(cell_type):
    n = 500
    y_true = rng.integers(0, 2, size=n)
    y_scores = np.clip(y_true * 0.6 + rng.normal(0, 0.3, size=n), 0, 1)
    return y_true, y_scores

# Data Loading (binn placeholder scores):
def load_binn_scores(cell_type):
    n = 500
    y_true = rng.integers(0, 2, size=n)
    y_scores = np.clip(y_true * 0.7 + rng.normal(0, 0.25, size=n), 0, 1)
    return y_true, y_scores

# Metrics (bootstrap roc logic):
def get_bootstrap_roc(y_true, y_scores):
    base_fpr = np.linspace(0, 1, 101)
    tprs, aucs = [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2: continue
        fpr, tpr, _ = roc_curve(y_true[idx], y_scores[idx])
        tprs.append(np.interp(base_fpr, fpr, tpr))
        aucs.append(auc(fpr, tpr))
    return base_fpr, np.mean(tprs, axis=0), np.percentile(tprs, [2.5, 97.5], axis=0), np.mean(aucs), np.std(aucs)

# Plotting (roc curve visualization):
def draw_roc(ax, cell_type):
    for label, loader, color in [("SVM", load_svm_scores, "steelblue"), ("BINN", load_binn_scores, "darkorange")]:
        y_t, y_s = loader(cell_type)
        fpr, m_tpr, ci, m_auc, s_auc = get_bootstrap_roc(y_t, y_s)
        ax.plot(fpr, m_tpr, color=color, label=f"{label} (AUC {m_auc:.2f}±{s_auc:.2f})")
        ax.fill_between(fpr, ci[0], ci[1], color=color, alpha=0.1)
    ax.plot([0, 1], [0, 1], "k--", lw=0.5)
    ax.set_title(f"ROC: {cell_type}")
    ax.legend(fontsize=7)

# Plotting (confusion matrix heatmap):
def draw_confusion(ax, cell_type):
    y_t, y_s = load_binn_scores(cell_type)
    cm = confusion_matrix(y_t, (y_s >= 0.5).astype(int), normalize="true")
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i,j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", color=color)
    ax.set_title(f"CM (BINN): {cell_type}")

# Orchestration (main figure generation):
def run_evaluation():
    for cell_type in CELL_TYPES:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        draw_roc(axes[0], cell_type)
        draw_confusion(axes[1], cell_type)
        plt.tight_layout()
        save_path = os.path.join(SAVE_DIR, f"eval_{cell_type}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Generated: {save_path}")

if __name__ == "__main__":
    run_evaluation()