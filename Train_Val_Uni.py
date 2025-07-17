import os
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from preprocessing.data_utils import AtomGraphDataset
from cgxas.models import CGXAS
from cgxas.trainer import GNNTrainer


def RSE(prediction, target):
    for i in range(int(len(prediction)/100)):
        if i==0:
            loss=torch.sum((target[i*100:(i+1)*100] - prediction[i*100:(i+1)*100])**2)/torch.sum((target[i*100:(i+1)*100])**2)
        else:
            loss+=torch.sum((target[i*100:(i+1)*100] - prediction[i*100:(i+1)*100])**2)/torch.sum((target[i*100:(i+1)*100])**2)
    return loss

def RSE2(prediction, target):
    for i in range(int(len(prediction)/100)):
        if i==0:
            loss=torch.sum((target[i*100:(i+1)*100] - prediction[i*100:(i+1)*100])**2)/torch.sum((target[i*100:(i+1)*100])**2)+torch.sum(((target[i*100+1:(i+1)*100]-target[i*100:(i+1)*100-1])- (prediction[i*100+1:(i+1)*100]-prediction[i*100:(i+1)*100-1]))**2)/torch.sum((target[i*100+1:(i+1)*100]-target[i*100:(i+1)*100-1])**2)
        else:
            loss+=torch.sum((target[i*100:(i+1)*100] - prediction[i*100:(i+1)*100])**2)/torch.sum((target[i*100:(i+1)*100])**2)+torch.sum(((target[i*100+1:(i+1)*100]-target[i*100:(i+1)*100-1])- (prediction[i*100+1:(i+1)*100]-prediction[i*100:(i+1)*100-1]))**2)/torch.sum((target[i*100+1:(i+1)*100]-target[i*100:(i+1)*100-1])**2)
    return loss


##Model Parameters
model_name = 'CGXAS_Uni'
# number of epochs in training
num_epochs = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device='cpu'
#learning rate 
lr =1e-3
# milestones to reduce learning rate in steps 
milestones = np.arange(10, 100, 10).tolist()

cgnn = CGXAS(
        n_node_feat=146, 
        n_hidden_feat=256,  
        out_feat=100, 
        conv_bias=True, 
        n_GNN=5, 
        n_MLP=5,
        node_activation="Sigmoid", 
        MLP_activation="Elu", 
        use_node_batch_norm=True, 
        use_edge_batch_norm=True, 
        cutoff=8, 
        n_grid_K=4, 
        n_Gaussian=64
        ).to(device)

if __name__ == '__main__':

    trainer = GNNTrainer(model=cgnn, 
                        model_name=model_name,
                        device=device,
                        metric_path="./best_model/metrics/metrics_"+model_name)
    ##Dataset Loading
    train_data = AtomGraphDataset(root='./processed_data/Site_XAS_all_train.pt.pt')
    val_data = AtomGraphDataset(root='./processed_data/Site_XAS_all_val.pt.pt')

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=True)

    optimizer = torch.optim.AdamW(cgnn.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                 milestones=milestones,
                                                 gamma=0.8)
    trainer.train_val(train_loader, val_loader, optimizer,RSE, scheduler, num_epochs, output_step=1)
