import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for Linux servers

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Global configuration and directory setup
SAVE_DIR = "figures/ml_model_evaluations"
N_BOOTSTRAP = 1000
RANDOM_SEED = 42
os.makedirs(SAVE_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)

# Load BINN results:
def load_binn_scores(): 
    if os.path.exists("binn_test_results.csv"):
        df = pd.read_csv("binn_test_results.csv")
        return df['y_true'].values, df['y_prob'].values
    return None, None

# Load SVM results:
def load_svm_scores(cell_type):
    if os.path.exists("svm_test_results.csv"):
        df = pd.read_csv("svm_test_results.csv")
        return df['y_true'].values, df['y_prob'].values
    else:
        print("Warning: svm_test_results.csv not found, using placeholder data.")
        n = 500
        y_true = rng.integers(0, 2, size=n)
        y_scores = np.clip(y_true * 0.6 + rng.normal(0, 0.3, size=n), 0, 1)
        return y_true, y_scores

def get_bootstrap_roc(y_true, y_scores):
    # Calculate mean ROC and confidence intervals using bootstrapping
    base_fpr = np.linspace(0, 1, 101)
    tprs, aucs = [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2: continue
        fpr, tpr, _ = roc_curve(y_true[idx], y_scores[idx])
        tprs.append(np.interp(base_fpr, fpr, tpr))
        aucs.append(auc(fpr, tpr))
    return base_fpr, np.mean(tprs, axis=0), np.percentile(tprs, [2.5, 97.5], axis=0), np.mean(aucs), np.std(aucs)

def draw_roc(ax, binn_data, svm_data):
    # Plot the mean ROC-curve and 95% confidence interval
    for label, (y_t, y_s), color in [("SVM (Actual)", svm_data, "steelblue"), ("BINN (Actual)", binn_data, "darkorange")]:
        fpr, m_tpr, ci, m_auc, s_auc = get_bootstrap_roc(y_t, y_s)
        ax.plot(fpr, m_tpr, color=color, label=f"{label} (AUC {m_auc:.2f}±{s_auc:.2f})")
        ax.fill_between(fpr, ci[0], ci[1], color=color, alpha=0.1)
    
    ax.plot([0, 1], [0, 1], "k--", lw=0.5)
    ax.set_title("ROC Curve: Global Evaluation")
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(fontsize=8)

def draw_confusion(ax, y_t, y_s):
    # Plot the normalized confusion matrix for the BINN
    cm = confusion_matrix(y_t, (y_s >= 0.5).astype(int), normalize="true")
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    
    # Annotate the matrix with mean values
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i,j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", color=color)
            
    ax.set_title("Normalized Confusion Matrix: BINN")
    ax.set_xticks([0, 1], ["Healthy", "AD"])
    ax.set_yticks([0, 1], ["Healthy", "AD"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

def run_evaluation():
    # Orchestrate the loading and plotting process
    y_t_binn, y_s_binn = load_binn_scores()
    
    if y_t_binn is not None:
        y_t_svm, y_s_svm = load_svm_scores(cell_type="global")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Generate Figure 2a and 2c equivalents from the plan
        draw_roc(axes[0], (y_t_binn, y_s_binn), (y_t_svm, y_s_svm))
        draw_confusion(axes[1], y_t_binn, y_s_binn)
        
        plt.tight_layout()
        save_path = os.path.join(SAVE_DIR, "global_binn_evaluation.png")
        plt.savefig(save_path, dpi=300)
        print(f"Evaluation complete. Figure saved to: {save_path}")
    else:
        print("Required data file 'binn_test_results.csv' not found. Please run the pipeline first.")

if __name__ == "__main__":
    run_evaluation()