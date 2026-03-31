from torch.utils.data import DataLoader
import custom_train_test_split
import anndata as ad
from anndata.experimental import AnnCollection
from anndata.experimental.pytorch import AnnLoader
import SingleCellDataset

ALL_CELLTYPES = [0,1,2,3,4,5,6,7,8]

def read_adata(indices: list, train_size=0.8):
    """
    Reads the training anndata, testing anndata and the collection they come from.
    Indicies indicate celltype. 
    """
    train_adata, test_adata, acollection = custom_train_test_split.pipeline(indices, train_size)
    return train_adata, test_adata, acollection

def data_concatenate(acollection : AnnCollection):
    """
    Concatenate the AnnCollection to a single AnnData object
    """
    adata_conc = ad.concat(acollection, label="cell_type_low_res")
    return adata_conc
    
def train_test_adatasplit(train_adata: ad.AnnData, test_adata: ad.AnnData):
    """
    Creates the train/test split for the input anndata
    """
    X_train = train_adata.X
    y_train = train_adata.obs["AD_status"]
    X_test = test_adata.X
    y_test = test_adata.obs["AD_status"]

    return X_train, y_train, X_test, y_test

def get_dataloaders(train_adata : ad.AnnData, test_adata : ad.AnnData, batch_size=64):
    """
    Extracts data from AnnData and returns train and test DataLoaders.
    """
    X_train, X_test, y_train, y_test = train_test_adatasplit(train_adata, test_adata)

    train_dataset = SingleCellDataset(X_train, y_train)
    test_dataset = SingleCellDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


## TESTING
print("Reading adata...")
train_adata, test_adata, acollection = read_adata(ALL_CELLTYPES)
print("Getting dataloaders...")
train_loader, test_loader = get_dataloaders(train_adata, test_adata)
print("Concatenating data...")
adata_conc = data_concatenate(acollection)

