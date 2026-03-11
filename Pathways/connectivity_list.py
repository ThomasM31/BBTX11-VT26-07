import pandas as pd
from pathlib import Path
import os
import argparse

def get_gene_pathway_conns(project_path: str) -> pd.DataFrame:
    # Read GMT-files
    # GMT files contain entries with:
    # pathway name | pathway ID | list of associated genes
    print("Reading gene to pathway relationships")

    fname_gmt = 'ReactomePathways.gmt'
    full_path_gmt = os.path.join(project_path, fname_gmt)

    edges = []
    with open(full_path_gmt, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            pathway_id = parts[1]

            # only keep entries related to human biology
            if pathway_id.startswith('R-HSA'):
                genes = parts[2:]
                for gene in genes:
                    edges.append({'child': gene, 'parent': pathway_id})

    df_gene_pathway = pd.DataFrame(edges)
    return df_gene_pathway

def get_pathway_pathway_conns(project_path: str) -> pd.DataFrame:
    # Reading relation file 
    # This contains entries with child pathway | parent pathway
    print("Reading pathway to pathway relationships")

    fname_txt = 'ReactomePathwaysRelation.txt'
    full_path_txt = os.path.join(project_path, fname_txt)

    df_pathway_pathway = pd.read_csv(fname_txt, sep='\t', header=None)

    # only keep entries related to human biology
    df_pathway_pathway = df_pathway_pathway[df_pathway_pathway[0].str.startswith('R-HSA')]
    df_pathway_pathway.columns = ['parent', 'child']

    # Reverse child and parent so it matches format 
    # in the gene to pathway connections
    df_pathway_pathway = df_pathway_pathway[['child', 'parent']]
    
    return df_pathway_pathway
    
def make_connectivity_list(df_gene_pathway: pd.DataFrame, 
                           df_pathway_pathway: pd.DataFrame) -> pd.DataFrame:
    # Append gene to pathway and pathway to pathway rows
    print("Creating connectivity list")
    full_network = pd.concat([df_gene_pathway, df_pathway_pathway], ignore_index=True)
    return full_network

def save(project_path: str, 
        full_network: pd.DataFrame, 
        save_network: bool) -> None:

    if save_network:
        fname_save = 'binn_connectivity.csv'
        full_path_save = os.path.join(project_path, fname_save)
        print(full_path_save)
        
        full_network.to_csv(full_path_save, index=False)

        n = len(full_network)
        print(f'{fname_save} created with {n} connections.')
    else:
        print(f'File not saved as script was used with --no-save or save_data=False.')

def run(save_data: bool = True) -> None:
    project_path = Path('/data/shared/alzgene26/PathwayData/')

    # extracts gene to pathway connections
    df_gene_pathway = get_gene_pathway_conns(project_path)

    # extracts pathway to pathway connections
    df_pathway_pathway = get_pathway_pathway_conns(project_path)

    # appends gene to pathway connections and
    # pathway to pathway connections
    full_network = make_connectivity_list(df_gene_pathway, df_pathway_pathway)

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