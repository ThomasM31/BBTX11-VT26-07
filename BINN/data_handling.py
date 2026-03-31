from preprocessing import custom_train_test_split
import anndata as ad

def read_adata(indices: list, train_size=0.8):
    train_adata, test_adata, acollection = custom_train_test_split.pipeline(indices, train_size)
    return train_adata, test_adata, acollection
    
def train_test_adatasplit(train_adata: ad.AnnData, test_adata: ad.AnnData):
    X_train = train_adata.X
    y_train = train_adata.obs["AD_status"]
    X_test = test_adata.X
    y_test = test_adata.obs["AD_status"]

    return X_train, y_train, X_test, y_test

