import pandas as pd
from pathlib import Path
import os
import argparse

def read_gmt(project_path: str) -> pd.DataFrame:
    # Läs in GMT-filen (Gener -> Pathways)
    print("Läser in GMT-filen...")

    fname_gmt = 'ReactomePathways.gmt'

    full_path_gmt = os.path.join(project_path, fname_gmt)

    edges = []
    with open(full_path_gmt, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            pathway_id = parts[1]
            genes = parts[2:]
            for gene in genes:
                edges.append({'child': gene, 'parent': pathway_id})

    df_gene_pathway = pd.DataFrame(edges)
    return df_gene_pathway
    
def make_connectivity_list(project_path: str, df_gene_pathway: pd.DataFrame) -> pd.DataFrame:
    # Läs in Relations-filen (Pathway -> Parent Pathway)
    print("Läser in hierarkin...")

    fname_txt = 'ReactomePathwaysRelation.txt'
    full_path_txt = os.path.join(project_path, fname_txt)

    df_pathway_relation = pd.read_csv(fname_txt, sep='\t', header=None)
    df_pathway_relation.columns = ['parent', 'child']

    # Vi vänder på dem så de matchar 'child' -> 'parent' formatet
    df_pathway_relation = df_pathway_relation[['child', 'parent']]

    # Slå ihop allt till en stor konnektivitetslista
    print("Bygger den slutgiltiga matrisen...")
    full_network = pd.concat([df_gene_pathway, df_pathway_relation], ignore_index=True)
    return full_network

def save(project_path: str, full_network: pd.DataFrame, save_network: bool) -> None:
    fname_save = 'binn_connectivity.csv'
    full_path_save = os.path.join(project_path, fname_save)
    print(full_path_save)

    if save_network == True:
        full_network.to_csv(full_path_save, index=False)
        print(f"Klart! Filen 'binn_connectivity.csv' har skapats med {len(full_network)} kopplingar.")
    else:
        print(f"Klart! Ingen fil sparades, ty Save_Network var satt till False.")

def run(save_data: bool = True) -> None:
    project_path = Path('/data/shared/alzgene26/PathwayData/')

    df_gene_pathway = read_gmt(project_path)

    full_network = make_connectivity_list(project_path, df_gene_pathway)

    save(project_path, full_network, save_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a .gmt file with pathways to .csv file with connectivities."
    )

    # type --no-save when calling the file to make save_data False
    parser.add_argument(
        "--no-save", 
        dest="save_data", 
        action="store_false", 
        help="Disables saving the data (Default: True)"
    )
    
    parser.set_defaults(save_data=True)

    args = parser.parse_args()

    # Access the variable from the args object
    run(args.save_data)