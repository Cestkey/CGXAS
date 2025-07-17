import os.path
import json
import scipy
from scipy.special import jn_zeros,jn,sph_harm
import numpy as np
import pandas as pd
import glob
import torch
#from torch.utils.data import Dataset
from torch_geometric.data import Data, Dataset
from torch_geometric.data.collate import collate
import tqdm


class AtomGraphDataset(Dataset):
    """
    The dataset for XAS prediction. It contains the atomgraph of materials and their corresponding xas spectrum.
    """     
    def __init__(self, root):
        """
        Args:
            root (str): The path to the processed dataset.
        """        
        self.root = root
           
        if os.path.exists(self.root):
            self.graph_data = torch.load(self.root)
        else:
            self.graph_data=[]           
    
    def collate(data_list):

        if len(data_list) == 1:
            return data_list[0], None

        data, slices, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=False,
            add_batch=False,
        )

        return data, slices
    
    def download(self):
        pass

    def __len__(self):
        return len(self.graph_data)
    
    def len(self):
        pass
    
    def get(self):
        pass

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.int32, np.int64)):
            data = self.graph_data[idx] 
        if isinstance(idx, (list, tuple, np.ndarray)):
            data = [self.graph_data[i] for i in list(idx)]
        if isinstance(idx, slice):
            idx = np.arange(idx.start, min(idx.stop, len(self), idx.step))
            data = [self.graph_data[i] for i in idx]   
        
        return data
                   


