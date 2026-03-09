# general imports
import sys
from tqdm import tqdm

# sklearn
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# anndata
import anndata as ad
from anndata.experimental import AnnCollection

# 
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt 

# Own files
from preprocessing import custom_train_test_split_modified

LABELS = ['astro', 'exc1', 'exc2', 'exc3', 'immune', 'inhi', 'oligo', 'opcs', 'vasc']

def read_adata(indices: list, train_size=0.8):
    train_adata, test_adata, collection = custom_train_test_split_modified.pipeline(indices, train_size)
    return train_adata, test_adata, collection
    
def xy_datasplit(train_adata: ad.AnnData, test_adata: ad.AnnData):
    X_train = train_adata.X
    y_train = train_adata.obs["AD_status"]
    X_test = test_adata.X
    y_test = test_adata.obs["AD_status"]

    return X_train, y_train, X_test, y_test


def baseline_model(train_adata : ad.AnnData, test_adata: ad.AnnData):
    X_train, y_train, X_test, y_test = xy_datasplit(train_adata, test_adata)

    # Support Vector Machine, icke-linjär
    clf_svm = SVC(C=1, probability=True, gamma=0.001, kernel='rbf', class_weight='balanced') 

    # PCA
    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print("Fitting to model now!")
    clf_svm.fit(X_train_pca, y_train)
    
    y_scores = clf_svm.predict_proba(X_test_pca)[:, 1]

    # ROC AUC 
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc = roc_auc_score(y_test, y_scores)
    print(f"AUC: {auc:.4f}")

    """
    # Plot & save ROC-curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})")
    plt.plot([0,1], [0,1], linestyle="--")  # random baseline
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("astro SVM ROC Curve")
    plt.legend()
    plt.savefig("/data/users/thomath/kand/curves_graphs/SVM astro roc_curve.png")
    """

    # CV
    print("Cross validating now!")
    scores = cross_val_score(clf_svm, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"CV Scores: {scores}")
    print(f"Mean CV AUC: {scores.mean():.4f}")

    y_pred = clf_svm.predict(X_test_pca)
    print(classification_report(y_test, y_pred))


# indices: 0=astro, 1=exc1, 2=exc2, 3=exc3, 4=immune, 5=inhi, 6=oligo, 7=opcs, 8=vasc
train_adata, test_adata, acollection = read_adata([0,1,2,3,4,5,6,7,8], train_size=0.8)
#baseline_model(train_adata, test_adata)

# AUC and Mean CV AUC
svm_dataset_performances = {"ALL": "AUC: 0.5612 Mean CV AUC: 0.6649",
                        "astro": "AUC: 0.6796, Mean CV AUC: 0.6597",
                        "astro_mod": "AUC: 0.6788, Mean CV AUC: 0.6597",
                        "exc1": "AUC: 0.6049, Mean CV AUC: 0.7031",
                        "exc2": "AUC: 0.6399, Mean CV AUC: 0.6682",
                        "exc3": "AUC: 0.6057, Mean CV AUC: 0.6246",
                        "immune": "AUC: 0.6026, Mean CV AUC: 0.6157",
                        "inhi": "AUC: 0.5569, Mean CV AUC: 0.5980",
                        "oligo": "AUC: 0.6375, Mean CV AUC: 0.6965",
                        "opcs": "AUC: 0.5826, Mean CV AUC: 0.6591",
                        "vasc": "AUC: 0.6457, Mean CV AUC: 0.5679", # NOTE: RED-FLAG (tur med test split)
                        }


## Classifiers
# Logistic Regression
# clf_lr = LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=0.1, solver="saga", max_iter=5000)
# Random Forest
# clf_rf = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=2, class_weight='balanced',random_state=42, n_jobs=-1)


# Parameter grid search
"""
clf_svm = SVC(probability=True, class_weight='balanced')

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf'] 
}

grid = GridSearchCV(clf_svm, param_grid, refit=True, cv=5, scoring='roc_auc')
print("CV-Searching parameter grid now!")
grid.fit(X_train, y_train)

print(f"Bästa parametrar: {grid.best_params_}")
"""


# PCA Elbow
"""
pca = PCA().fit(X_train) # Fit utan att sätta n_components för att se alla

plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axhline(y=0.90, color='r', linestyle='--', label='90% Varians')
plt.xlabel('Antal komponenter')
plt.ylabel('Kumulativ förklarad varians')
plt.title('Hitta "armbågen"')
plt.legend()
plt.savefig("/data/users/thomath/kand/pca_elbow.png")
plt.close()
"""

# PCA Variance plotting
"""
    plt.figure(figsize=(10, 7))
    for label, color, name in zip([0, 1], ['blue', 'red'], ['Kontroll', 'Alzheimer']):
        mask = (y_train == label)
        plt.scatter(X_train_pca[mask, 0], X_train_pca[mask, 1], 
                    c=color, label=name, alpha=0.6, edgecolors='w')

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.legend()
    plt.title('PCA: Separation mellan Kontroll och Alzheimer')
    plt.savefig("pca_scatter_plot.png")
    plt.close()
    print("PCA-plottar sparade som PNG-filer.")
    """