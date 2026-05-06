import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime as dt

def perform_shap(
        model: nn.Module, 
        X_train_tensor: torch.Tensor, 
        X_test_tensor: torch.Tensor,
        gene_names: list[str],
        figpath: Path
        ) -> None:
    
    device = next(model.parameters()).device
    
    model.eval()

    # Create the background dataset 
    background = X_train_tensor.to(device)

    # Select target patients to explain
    test_patients = X_test_tensor.to(device)
    
    print("Initializing explainer and calculating SHAP values...")
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(test_patients)

    print(f"SHAP values: {shap_values}")

    # inspect the format
    print("\nformat inspection:")

    if isinstance(shap_values, list):
        print("SHAP returned a list (one array per class). Taking the positive class.")
        shap_matrix = shap_values[1]
    else:
        print("SHAP returned a single numpy array.")
        shap_matrix = shap_values

    # Ensure it's a numpy array and on CPU
    if torch.is_tensor(shap_matrix):
        shap_matrix = shap_matrix.cpu().numpy()
    shap_matrix = np.squeeze(shap_matrix) 

    print(f"Shape of the SHAP Matrix: {shap_matrix.shape} -> (Patients, Genes)")

    patient_0_shaps = shap_matrix[0]
    patient_0_genes = test_patients[0].detach().cpu().numpy()

    df = pd.DataFrame({
        'Gene_Name': gene_names,
        'Raw_Gene_Expression': patient_0_genes,
        'SHAP_Impact': patient_0_shaps  
    })

    # Sort by absolute impact to find the top drivers
    df['Absolute_Impact'] = df['SHAP_Impact'].abs()
    df_sorted = df.sort_values(by='Absolute_Impact', ascending=False).drop(columns=['Absolute_Impact'])

    print("\nTop 5 Driving Genes for Patient 0:")
    print(df_sorted.head())

    print("\nGENERATING PLOTS...")

    # Extract the baseline expected value
    try:
        base_value = explainer.expected_value
    except AttributeError:
        print("Expected value not found in explainer. Calculating from background...")
        with torch.no_grad():
            base_value = model(background).mean(dim=0).cpu().numpy()
            base_value = base_value[0]
    
    # Build the SHAP explanation object manually
    shap_explanation = shap.Explanation(
        values=shap_matrix,                  
        base_values=base_value,              
        data=test_patients.cpu().numpy(),          
        feature_names=gene_names  
    )

    now = dt.now().strftime("%y%m%d_%H%M")


    # Save rawdata for figures:
    # E.g. the SHAP-matrix (importance for each gene per patient)
    shap_df = pd.DataFrame(shap_matrix, columns=gene_names)
    shap_df.to_csv(figpath / f'real_shap_values.csv', index=False)
    raw_expr_df = pd.DataFrame(test_patients.cpu().numpy(), columns=gene_names)
    raw_expr_df.to_csv(figpath / f'real_expression_values.csv', index=False)
    
    print(f"Data saved to CSV in: {figpath}")


    # generate the plots
    print("Displaying Beeswarm Plot...")
    shap.plots.beeswarm(shap_explanation, show=False)
    plt.savefig(figpath / f'beeswarm_{now}.png', bbox_inches='tight')
    plt.close()

    for i in list(range(3)):
        print(f"Displaying Waterfall Plot for Patient {i}...")
        shap.plots.waterfall(shap_explanation[i], show=False)
        plt.savefig(figpath / f'waterfall_{i}_{now}.png', bbox_inches='tight')
        plt.close()

    print("Displaying Violin Plot...")
    shap.plots.violin(shap_explanation, show=False)
    plt.savefig(figpath / f'violin_plot_{now}.png', bbox_inches='tight')
    plt.close()

    print(f"Plots saved to: \n {figpath}")