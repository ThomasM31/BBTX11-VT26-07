import pandas as pd
import numpy as np

def generate_data():
    # 1. Define a larger hierarchy
    # Layer 1: Genes (Inputs)
    genes_risk = ['APOE', 'TREM2', 'BIN1', 'ABCA7', 'SORL1']
    genes_prot = ['CD33', 'CLU', 'PICALM', 'CR1', 'EPHA1']
    genes_noise = [f'GENE_{i}' for i in range(10)] # 10 low-impact genes
    
    # Layer 2: Sub-pathways
    sub_pathways = ['Lipid_Metabolism', 'Microglial_Activation', 'Endocytosis', 'Autophagy', 'Complement_System']
    
    # Layer 3: Main Pathways
    main_pathways = ['Neuroinflammation', 'Protein_Clearance']

    # 2. Map Genes -> Sub-pathways
    # We'll assign high absolute weights to 'known' genes and low weights to 'noise' genes
    g2s_rows = []
    for g in genes_risk:
        g2s_rows.append({'Source': g, 'Target': np.random.choice(sub_pathways[:2]), 'SHAP_Weight': np.random.uniform(2, 5)})
    for g in genes_prot:
        g2s_rows.append({'Source': g, 'Target': np.random.choice(sub_pathways[2:4]), 'SHAP_Weight': np.random.uniform(-4, -1)})
    for g in genes_noise:
        g2s_rows.append({'Source': g, 'Target': np.random.choice(sub_pathways), 'SHAP_Weight': np.random.uniform(-0.5, 0.5)})
    
    gene_to_sub = pd.DataFrame(g2s_rows)

    # 3. Map Sub-pathways -> Main Pathways
    s2m_rows = []
    # Most sub-pathways feed into Neuroinflammation or Protein Clearance
    for s in sub_pathways:
        target = 'Neuroinflammation' if 'Micro' in s or 'Comp' in s else 'Protein_Clearance'
        # Scale weight based on incoming gene weights (summing them roughly)
        s2m_rows.append({'Source': s, 'Target': target, 'SHAP_Weight': np.random.uniform(-3, 6)})
    
    sub_to_main = pd.DataFrame(s2m_rows)

    # 4. Map Main Pathways -> Output
    m2o_rows = [
        {'Source': 'Neuroinflammation', 'Target': 'Alzheimers_Status', 'SHAP_Weight': 7.5},
        {'Source': 'Protein_Clearance', 'Target': 'Alzheimers_Status', 'SHAP_Weight': -5.0}
    ]
    
    main_to_out = pd.DataFrame(m2o_rows)

    # Combine
    return pd.concat([gene_to_sub, sub_to_main, main_to_out], axis=0).reset_index(drop=True)
