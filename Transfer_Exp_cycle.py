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

def find_E0_simple(energy, intensity):
    
    index=np.where(intensity>=0.5*np.mean(intensity[-10:]))[0]
    for i in range(0,len(index)-10):
        if index[i]+10==index[i+10]:
            id=index[i]
            break
    E0 = energy[id]

    return E0

def predict(graph, model,device='cpu'):
    """
    XAS prediction of the crystal graph using the given model
    
    return: a numpy array of predicted spectrum
    """   
    graph = graph.to(device)
    nodes,edge_sources,edge_targets,edge_distance,mask,combine_sets,plane_wave = graph.x,graph.edge_sources,graph.edge_targets,graph.edge_distance,graph.mask,graph.combine_sets,graph.plane_wave
    batch_seg = torch.tensor(np.repeat(0, nodes.shape[0]), device=device,dtype=torch.int64)

    model.to(device)
    model.eval()
    with torch.no_grad():
        graph_pred = model(nodes,edge_sources,edge_targets,edge_distance,mask,batch_seg,combine_sets,plane_wave)
        graph_pred = graph_pred.cpu().numpy().flatten()
    return graph_pred

def interpolate_curve(x,y,x_new,type):
    """
    Interpolate curve with specific range.
    Args:
    x (array_like): x value of the curve.
    y (array_like): y value of the curve.
    x_new (array_like): x value of points to be interpolated.
    type (str): ways to interpolate the curve. "linear": linear interpolation. "B-spline": B-spline interpolation.

    Return:
    y_new (array_like): y value of points be interpolated. 
    """
    from scipy import interpolate 
    if min(x_new)<min(x):
        xs=np.linspace(min(x_new),min(x),10)
        ys=np.zeros(10)
        x=np.append(xs,x)
        y=np.append(ys,y)
    if max(x_new)>max(x):
        xl=np.linspace(max(x),max(x_new),10)
        yl=np.ones(10)
        x=np.append(x,xl)
        y=np.append(y,yl)   
    if type=="linear":
        f_linear = interpolate.interp1d(x, y)
        y_new=f_linear(x_new)
        return y_new
    elif type=="B-spline":
        tck = interpolate.splrep(x, y)
        y_new = interpolate.splev(x_new, tck)
        return y_new
    else:
        print("Illegal type! Choose linear or B-spline")
        return

def data_calibrate(dataset_loc,dataset_cali_loc,model,device='cpu'):
    """
   Calibrate the dataset using the given model
    """   
    data=torch.load(dataset_loc)
    for i in range(0,len(data)):
        if device=='cpu':
            E0_1=find_E0_simple(intensity=data[i].spectrum_i.numpy(), energy=data[i].spectrum_e.numpy())
            pred=predict(data[i],model,device)
            E0_2=find_E0_simple(intensity=pred, energy=data[i].spectrum_e.numpy())  
            x=data[i].spectrum_e.numpy()+(E0_2-E0_1)
            y=data[i].spectrum_i.numpy()
            x_new=data[i].spectrum_e.numpy()
            y_new=interpolate_curve(x,y,x_new,'linear')
        else:
            E0_1=find_E0_simple(intensity=data[i].spectrum_i.cpu().numpy(), energy=data[i].spectrum_e.cpu().numpy())
            pred=predict(data[i],model,device)
            E0_2=find_E0_simple(intensity=pred, energy=data[i].spectrum_e.cpu().numpy())              
   
                  
            x=data[i].spectrum_e.cpu().numpy()+(E0_2-E0_1)
            y=data[i].spectrum_i.cpu().numpy()
            x_new=data[i].spectrum_e.cpu().numpy()
            y_new=interpolate_curve(x,y,x_new,'linear')

                
        x_new=x_new.reshape(-1,1)
        y_new=y_new.reshape(-1,1)
        data[i].spectrum_e=torch.Tensor(x_new.reshape(-1))
        data[i].spectrum_i=torch.Tensor(y_new.reshape(-1))
        torch.save(data, dataset_cali_loc)    
  

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

    #Initial Transfer learning on the experimental data
    cgnn.load_state_dict(torch.load('./best_model/CGXAS_Uni.pt'))
    model_name = 'CGXAS_Exp_S_cyle0'
    train_data = torch.load('./processed_data/Exp_S_train.pt')
    val_data = torch.load('./processed_data/Exp_S_val.pt')       
    train_loader = DataLoader(train_data, batch_size=6, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=6, shuffle=True)

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
        trainer.train_val(train_loader, val_loader, optimizer,RSE, scheduler, 30, output_step=1,start_epoch=k*30,save_temp=False)
        cgnn.load_state_dict(torch.load('./best_model/%s.pt'%model_name))
    
    for cycle in range(1,10):
        #calibrate the dataset
        data_calibrate(dataset_loc='./processed_data/Site_XAS_S_train.pt',dataset_cali_loc='./processed_data/Site_XAS_S_train_cali_%i.pt'%cycle,model=cgnn,device=device)
        data_calibrate(dataset_loc='./processed_data/Site_XAS_S_val.pt',dataset_cali_loc='./processed_data/Site_XAS_S_val_cali_%i.pt'%cycle,model=cgnn,device=device)

        #Transfer learning on the calibrated simulated data
        cgnn.load_state_dict(torch.load('./best_model/CGXAS_Uni.pt'))
        model_name = 'CGXAS_S_cali%i'%cycle
        train_data = torch.load('./processed_data/Site_XAS_S_train_cali_%i.pt'%cycle)
        val_data = torch.load('./processed_data/Site_XAS_S_val_cali_%i.pt'%cycle)       
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=128, shuffle=True)


        for i in cgnn.parameters():
            i.requires_grad=False
        for i in cgnn.MLP.parameters():
            i.requires_grad=True

        trainer = GNNTrainer(model=cgnn, 
                        model_name=model_name,
                        device=device,
                        metric_path="./best_model/metrics/metrics_"+model_name)
        optimizer = torch.optim.AdamW(cgnn.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones,gamma=0.8)
        trainer.train_val(train_loader, val_loader, optimizer,RSE, scheduler, 100, output_step=1,start_epoch=0,save_temp=False)    

        #Transfer learning on the experimental data
        cgnn.load_state_dict(torch.load('./best_model/CGXAS_S_cali%i.pt'%cycle))
        model_name = 'CGXAS_Exp_S_cycle%i'%cycle
        train_data = torch.load('./processed_data/Exp_S_train.pt')
        val_data = torch.load('./processed_data/Exp_S_train.pt')       
        train_loader = DataLoader(train_data, batch_size=6, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=6, shuffle=True)


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
            trainer.train_val(train_loader, val_loader, optimizer,RSE, scheduler, 30, output_step=1,start_epoch=k*30,save_temp=False)
            cgnn.load_state_dict(torch.load('./best_model/%s.pt'%model_name))           


