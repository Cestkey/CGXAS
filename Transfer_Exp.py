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

def data_split(ndata,ntrain,nval,ntest,seed):
    random_state = np.random.RandomState(seed=seed)
    all_idx = np.arange(ndata)
    all_idx = random_state.permutation(all_idx)
    idxs = {'train' : all_idx[:ntrain],'val' : all_idx[ntrain : ntrain+nval],'test' : all_idx[ntrain+nval:ntrain+nval+ntest]}
    return idxs

##Model Parameters
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
    for count in range(10):
        cgnn.load_state_dict(torch.load('./best_model/CGXAS_Uni.pt'))
        model_name = 'CGXAS_Exp_S_%i'%count
        data= AtomGraphDataset(root='./processed_data/Exp_S.pt')
        idxs=data_split(ndata=48,ntrain=36,nval=6,ntest=6,seed=count+10)
        train_data = [data.graph_data[i] for i in idxs['train']]
        val_data = [data.graph_data[i] for i in idxs['val']]
        test_data = [data.graph_data[i] for i in idxs['test']]
        #train_data = torch.load('./processed_data1/Exp_input_146_S_all_2_train_%i.pt'%count)
        #val_data = torch.load('./processed_data1/Exp_input_146_S_all_2_val_%i.pt'%count)       
        train_loader = DataLoader(train_data, batch_size=6, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=6, shuffle=True)
        torch.save(train_data, './processed_data1/Exp_S_train_%i.pt'%count)
        torch.save(val_data, './processed_data1/Exp_S_val_%i.pt'%count)
        torch.save(test_data, './processed_data1/Exp_S_test_%i.pt'%count)

        for k in [0,1,2,3,4]:
            for i in cgnn.parameters():
                i.requires_grad=False
            for i in cgnn.MLP[k].parameters():
                i.requires_grad=True

            trainer = GNNTrainer(model=cgnn, 
                        model_name=model_name,
                        device=device,
                        metric_path="./best_model/metrics/metrics_"+model_name)
            optimizer = torch.optim.AdamW(cgnn.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones,gamma=0.8)
            trainer.train_val(train_loader, val_loader, optimizer,RSE2, scheduler, 30, output_step=1,start_epoch=k*30,save_temp=False)
            cgnn.load_state_dict(torch.load('./best_model/%s.pt'%model_name))
