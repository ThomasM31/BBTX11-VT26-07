import custom_train_test_split
import anndata as ad
from anndata.experimental import AnnCollection, AnnLoader
from binn_training import *

def read_adata(indices: list, 
               filepath: str, 
               train_size=0.8) -> tuple[ad.AnnData, ad.AnnData, AnnCollection]:
    """
    Reads the training anndata, testing anndata and the collection they come from.
    Indicies indicate celltype. 
    """
    train_adata, test_adata, acollection = custom_train_test_split.pipeline(indices, filepath, train_size)
    return train_adata, test_adata, acollection

def data_concatenate(acollection : AnnCollection) -> ad.AnnData:
    """
    Concatenate the AnnCollection to a single AnnData object
    """
    adata_conc = ad.concat(acollection, label="cell_type_low_res")
    return adata_conc
    
def train_test_adatasplit(train_adata: ad.AnnData, 
                          test_adata: ad.AnnData):
    """
    Creates the train/test split for the input anndata
    """
    X_train = train_adata.X
    y_train = train_adata.obs["AD_status"]
    X_test = test_adata.X
    y_test = test_adata.obs["AD_status"]

    return X_train, y_train, X_test, y_test

def create_dataloaders(train_adata: ad.AnnData,
                       test_adata: ad.AnnData,
                       batch_size=16) -> tuple[ad.AnnData, ad.Anndata]:
    """
    Create dataloaders using AnnLoader
    """
    
    train_loader = AnnLoader(train_adata, batch_size, shuffle=True)
    test_loader = AnnLoader(test_adata, batch_size, shuffle=False)
    return train_loader, test_loader