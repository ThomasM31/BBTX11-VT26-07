import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime as dt
import torch.nn.functional as F
import logging

def perform_shap(
        model: nn.Module, 
        X_train_tensor: torch.Tensor, 
        X_test_tensor: torch.Tensor,
        gene_names: list[str],
        figpath: Path,
        stage: str,
        date: str
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
    shap_df.to_csv(figpath / f'real_shap_values_{date}.csv', index=False)
    raw_expr_df = pd.DataFrame(test_patients.cpu().numpy(), columns=gene_names)
    raw_expr_df.to_csv(figpath / f'real_expression_values_{date}.csv', index=False)
    
    print(f"Data saved to CSV in: {figpath}")

    # generate the plots
    print("Displaying Beeswarm Plot...")
    shap.plots.beeswarm(shap_explanation, show=False, max_display=11)
    plt.savefig(figpath / f'beeswarm_{stage}_{date}.png', bbox_inches='tight')
    plt.close()

    for i in list(range(3)):
        print(f"Displaying Waterfall Plot for Patient {i}...")
        shap.plots.waterfall(shap_explanation[i], show=False, max_display=11)
        plt.savefig(figpath / f'waterfall_{stage}_{date}_{i}.png', bbox_inches='tight')
        plt.close()

    print("Displaying Violin Plot...")
    shap.plots.violin(shap_explanation, show=False, max_display=11)
    plt.savefig(figpath / f'violin_plot_{stage}_{date}.png', bbox_inches='tight')
    plt.close()

    print(f"Plots saved to: \n {figpath}")


def layerwise_shap(model, X_train_tensor, X_test_tensor, masks, device):
    model.eval()
    all_edges = []
    
    # 1. Keep the whole chain connected
    current_x = X_test_tensor.clone().detach().to(device).requires_grad_(True)
    
    activations = []
    
    # 2. Forward Pass WITHOUT detaching
    for i, layer in enumerate(model.model_layers):
        # Instead of detaching, we keep the current_x in the chain
        # and tell PyTorch to save its gradient during backward
        current_x.retain_grad() 
        activations.append(current_x)

        mask = getattr(model, f'mask_{i}')
        masked_weight = layer.weight * (mask.t() if mask.shape != layer.weight.shape else mask)

        # Compute next layer
        current_x = F.linear(current_x, masked_weight, layer.bias)
        
        if i < len(model.model_layers) - 1:
            current_x = model.layer_norms[i](current_x)
            current_x = model.activation_fn(current_x)
            current_x = model.dropout(current_x)

    # 3. Backward Pass
    model.zero_grad()
    current_x.sum().backward()

    # 4. Importance Calculation
    for i in range(len(masks)):
        mask_key = f'df{i}'
        current_mask = masks[mask_key]
        
        # --- RESTORED VARIABLES ---
        sources = current_mask.index.tolist()
        targets = current_mask.columns.tolist()
        # --------------------------

        act = activations[i]
        if act.grad is not None:
            # Importance = Mean absolute value of (Activation * Gradient)
            #importance_scores = (act * act.grad).abs().mean(dim=0).cpu().detach().numpy()
            importance_scores = (act * act.grad).mean(dim=0).cpu().detach().numpy()
        else:
            logging.warning(f"No gradients for layer {i}!")
            importance_scores = np.zeros(len(sources))

        # Map to Sankey DataFrame
        for s_idx, source_name in enumerate(sources):
            weight = importance_scores[s_idx]
            
            # Find active connections in the adjacency mask
            active_targets_mask = current_mask.iloc[s_idx] != 0
            active_target_names = current_mask.columns[active_targets_mask].tolist()
            
            for target_name in active_target_names:
                all_edges.append({
                    'Source': source_name,
                    'Target': target_name,
                    'SHAP_Weight': float(weight),
                    'Source_Layer': i,
                    'Target_Layer': i + 1
                })

    return pd.DataFrame(all_edges)