# sklearn
import datetime
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import shap
import sys

# anndata
import anndata as ad
from anndata.experimental import AnnCollection
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Own files
import BINN.custom_train_test_split as ctts
import BINN.data_handling as dh

import pipeline_paths as ppaths

pp = ppaths.PipelinePaths(True, 'mg_200_mc_200_mhvg1000')
result_save_path = pp.svm_results_path
fig_save_path = pp.svm_results_path
data_path = pp.compl_full_pipe_path
mask_path = pp.mask_full_pipe_path
MASK_PATHS = [mask_path / f"oligo_exc3_exc2_vasc_immune_astro_inhi_opcs_exc1_layer_{i}_mask.csv" 
            for i in range(5)]

# GLOBALS
LABELS = ['astro', 'exc1', 'exc2', 'exc3', 'immune', 'inhi', 'oligo', 'opcs', 'vasc']
ALL_CELLTYPES = [0,1,2,3,4,5,6,7,8]
TRAIN_SIZE = 0.8

def read_adata(indices: list, train_size=0.8):
    train_adata, test_adata, collection = ctts.pipeline(indices, data_path, train_size)
    return train_adata, test_adata, collection
    
def xy_datasplit(train_adata: ad.AnnData, test_adata: ad.AnnData):
    X_train = train_adata.X
    y_train = train_adata.obs["AD_status"]
    X_test = test_adata.X
    y_test = test_adata.obs["AD_status"]

    return X_train, y_train, X_test, y_test

import scipy.sparse

def baseline_model(train_adata : ad.AnnData, test_adata: ad.AnnData):
    X_train, y_train, X_test, y_test = xy_datasplit(train_adata, test_adata)

    if scipy.sparse.issparse(X_train):
        X_train = X_train.toarray()
    if scipy.sparse.issparse(X_test):
        X_test = X_test.toarray()

    print("Creating model...")
    from sklearn.preprocessing import StandardScaler
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('svm', SVC(kernel='linear', probability=True, class_weight='balanced', C=1))
    ])

    print("Fitting to model now!")
    pipeline.fit(X_train, y_train)
    
    y_scores = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    # ROC AUC 
    fpr, tpr, _ = roc_curve(y_test, y_scores, pos_label=1)
    auc = roc_auc_score(y_test, y_scores)
    print(f"AUC: {auc:.4f}")
    
    # CV
    print("Cross validating now!")
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"CV Scores: {scores}")
    print(f"Mean CV AUC: {scores.mean():.4f}")

    df_svm = pd.DataFrame({'y_true': y_test, 'y_prob': y_scores})
    df_svm.to_csv("svm_test_results.csv", index=False)
    print("Saved: svm_test_results.csv")

    # Accuracy 
    acc = accuracy_score(y_test, y_pred)
    print(f"accuracy for svm: {acc}")

    # Confusion Matrix
    print("Saving confusion matrix..")
    os.makedirs(SAVE_DIR, exist_ok=True) 
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    cmap = plt.get_cmap('Blues')
    cmd = ConfusionMatrixDisplay(cm, display_labels=["Frisk", "AD"])
    cmd.plot(cmap=cmap)#.figure_.savefig('confusion_matrix_SVM.png')

    plt.title("Confusion Matrix (SVM)")
    plt.savefig(fig_save_path / f'confusion_matrix_SVM_{date}.png')
    print(f'Saved CM to {fig_save_path / f'confusion_matrix_SVM_{date}.png'}')
    plt.close()
    print(f"Figure saved: {save_path}")

    # Classification report 
    report = classification_report(y_test, y_pred, output_dict=True)
    print(report)

    print("\n--- Starting SHAP Analysis ---")
    gene_names = train_adata.var_names.tolist()

    print("Scaling data for SHAP...")
    X_train_scaled = pipeline.named_steps['scaler'].transform(X_train)
    X_test_scaled = pipeline.named_steps['scaler'].transform(X_test)

    print("Initializing LinearExplainer...")
    # We pass the scaled background data directly
    explainer = shap.LinearExplainer(pipeline.named_steps['svm'], X_train_scaled)
    
    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(X_test_scaled)
    
    # Handle SHAP output format (LinearExplainer sometimes returns lists, sometimes arrays)
    if isinstance(shap_values, list):
        shap_matrix = shap_values[1]
    else:
        shap_matrix = shap_values

    print("Packaging data for modern beeswarm plot...")
    
    # Safely extract the base value (expected average prediction)
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
    else:
        base_value = explainer.expected_value

    # Package everything into a modern SHAP Explanation object
    shap_explanation = shap.Explanation(
        values=shap_matrix,                  
        base_values=base_value,              
        data=X_test_scaled,          
        feature_names=gene_names  
    )

    print("Generating SHAP beeswarm plot...")
    shap.plots.beeswarm(shap_explanation, show=False, max_display=11)
    
    shap_save_path = os.path.join(SAVE_DIR, 'shap_beeswarm_SVM' + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S") + '.png')
    plt.savefig(shap_save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"SHAP figure saved: {shap_save_path}")
    print("Baseline model complete!")    
    

# GLOBALS
LABELS = ['astro', 'exc1', 'exc2', 'exc3', 'immune', 'inhi', 'oligo', 'opcs', 'vasc']
ALL_CELLTYPES = [0,1,2,3,4,5,6,7,8]
base_path = "/data/shared/alzgene26/data"
data_path = base_path + "/processed_data/completed/full_pipeline/mg_200_mc_200_mhvg1000/"
MASK_PATHS = [f"/data/shared/alzgene26/PathwayData/MaskMatrixLayers/full_pipeline/mg_200_mc_200_mhvg1000/oligo_exc3_exc2_vasc_immune_astro_inhi_opcs_exc1_layer_{i}_mask.csv" 
              for i in range(5)]
TRAIN_SIZE = 0.8
SAVE_DIR = "figures/ml_model_evaluations"


# Pipeline -------------------------------------------------------------------
print("Reading processed adata...")
datasets = ctts.read_files(to_include=ALL_CELLTYPES, filepath=data_path)
#datasets = ctts.read_files(to_include=[8], filepath=data_path)
print("Dataset rollup...")
patient_datasets = dh.rollup_to_patient_level(datasets)
print("Reading masks...")
masks = dh.read_masks(MASK_PATHS, print_shapes=True)
print("Aligning adatas to BINN...")
datasets_aligend = dh.subset_genes(patient_datasets, masks['df0'])
print("Padding adatas to BINN-ready shape...")
datasets_padded = dh.pad_align_data(datasets_aligend, masks["df0"])
print("Starting Global Rollup with missing subject handling...")
adata_global = dh.create_global_with_missing_patients(datasets_padded)
print("Creating train/test split...")
train_adata, test_adata = ctts.custom_train_test_split(adata_global, train_size=TRAIN_SIZE)
print("Running baseline model...")
baseline_model(train_adata, test_adata)
# -------------------------------------------------------------------

# AUC and Mean CV AUC
svm_dataset_performances = {"ALL": "AUC: 0.5641 Mean CV AUC: 0.8033", 
                        "astro": "AUC: 0.6043, Mean CV AUC: 0.6626",
                        "exc1": "AUC: 0.6112, Mean CV AUC: 0.6885",
                        "exc2": "AUC: 0.5741, Mean CV AUC: 0.6808",
                        "exc3": "AUC: 0.6589, Mean CV AUC: 0.6384",
                        "immune": "AUC: 0.6932, Mean CV AUC: 0.6577",
                        "inhi": "AUC: 0.6067, Mean CV AUC: 0.6783",
                        "oligo": "AUC: 0.5875, Mean CV AUC: 0.7042",
                        "opcs": "AUC: 0.6158, Mean CV AUC: 0.6731",
                        "vasc": "AUC: 0.6630, Mean CV AUC: 0.5699",
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
