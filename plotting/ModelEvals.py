import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for Linux servers

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy.typing as npt
from typing import Tuple, Optional

import pipeline_paths as ppaths

# Global configuration and directory setup
pp = ppaths.PipelinePaths(True)
SAVE_DIR = pp.figures_path / 'model_eval'
N_BOOTSTRAP = 1000
RANDOM_SEED = 42
os.makedirs(SAVE_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)


def load_scores(path: str, mode_binary: bool) -> Tuple[Optional[npt.NDArray[np.int_]], Optional[npt.NDArray[np.float64]]]:
    '''
    Loads binary or probability prediction values. 
    Binary prediction values are used for Confusion Matrix.
    Probability prediction values are used for ROC.
    '''
    try:
        df = pd.read_csv(path)
        y_true = df['y_true'].to_numpy(dtype=np.int_)
        if mode_binary:
            if 'y_pred' not in df.columns: df['y_pred'] = df['y_prob'].round()
            y_prob = df['y_pred'].to_numpy(dtype=np.float64)
        else:
            y_prob = df['y_prob'].to_numpy(dtype=np.float64)
        return y_true, y_prob
    except FileNotFoundError:
        print(f'File {path} not found!')
        
        return (
            np.concatenate((np.ones((5,1), dtype=int), np.zeros((5,1), dtype=int))), 
            np.concatenate((np.zeros((5,1)), np.ones((5,1))))
            )

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
    for label, (y_t, y_s), color in [("SVM", svm_data, "steelblue"), ("BINN", binn_data, "darkorange")]:
        fpr, m_tpr, ci, m_auc, s_auc = get_bootstrap_roc(y_t, y_s)
        ax.plot(fpr, m_tpr, color=color, label=f"{label} (AUC {m_auc:.2f}±{s_auc:.2f})")
        ax.fill_between(fpr, ci[0], ci[1], color=color, alpha=0.1)
    
    ax.plot([0, 1], [0, 1], "k--", lw=0.5)
    ax.set_title("ROC-curve: Global evaluation")
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(fontsize=8)

def draw_confusion(ax, y_t:npt.NDArray[np.int_], y_s: npt.NDArray[np.float64], label:str):
    # Plot the normalized confusion matrix for the BINN
    cm = confusion_matrix(y_t, y_s, normalize="true")
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    
    # Annotate the matrix with mean values
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i,j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", color=color)
            
    ax.set_title(f"Normalized confusion matrix: {label}")
    ax.set_xticks([0, 1], ["Healthy", "AD"])
    ax.set_yticks([0, 1], ["Healthy", "AD"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

def run_roc(binn_path: str, svm_path: str, labels: list[str] = ['']):
    # Orchestrate the loading and plotting process
    y_t_binn, y_s_binn = load_scores(binn_path, False)

    if y_t_binn is not None:
        y_t_svm, y_s_svm = load_scores(svm_path, False)
        
        fig, axes = plt.subplots(1, 1)
        
        draw_roc(axes, (y_t_binn, y_s_binn), (y_t_svm, y_s_svm))
        
        plt.tight_layout()
        save_path = os.path.join(SAVE_DIR, "global_binn_evaluation_en.png")
        plt.savefig(save_path, dpi=300)
        print(f"Evaluation complete. Figure saved to: {save_path}")
    else:
        print(f"File {binn_path} not found, or required column(s) missing.")


def run_CM(in_path: str, out_path: str, label: str) -> None:
    # Orchestrate the loading and plotting process
    y_t_binn, y_s_binn = load_scores(in_path, True)    

    fig, ax = plt.subplots(1, 1)
    
    draw_confusion(ax, y_t_binn, y_s_binn, label)
    
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, out_path)
    plt.savefig(save_path, dpi=300)
    print(f"Evaluation complete. Figure saved to: {save_path}")
    plt.tight_layout()

if __name__ == "__main__":

    binn_path = '/data/shared/alzgene26/data/results/binn_results/binn_test_results_260508_0940.csv'
    svm_path = '/data/shared/alzgene26/data/results/svm_results/svm_test_results_260512_0806.csv'
    gen_path = '/data/shared/alzgene26/data/results/binn_results/gen_test_results_260508_0940_conv.csv'
    run_roc(binn_path, svm_path)
    
    run_CM(svm_path, 'svm_evaluation_en_260512_0806.png', 'SVM (ROSMAP)')
    run_CM(binn_path, 'binn_evaluation_en.png', 'BINN (ROSMAP)')
    run_CM(gen_path, 'swdbb_evaluation_en.png', 'BINN (SWDBB)')