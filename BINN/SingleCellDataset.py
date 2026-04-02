import anndata as ad, scipy as sp
import torch

class SingleCellDataset(ad.AnnData):
    """
    Custom Pytorch dataset for anndata objects, converts to Pytorch format
    """
    def __init__(self, X, y):
        self.X = X
        self.y = torch.LongTensor(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Convert sparse row to dense tensor on the fly to save memory
        row = self.X[idx]
        if sp.issparse(row):
            row = row.toarray().squeeze()
        return torch.FloatTensor(row), self.y[idx]
    
## Must X also be tensors??????