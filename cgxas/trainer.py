from typing import Any
import torch
import numpy as np
import pandas as pd
from cgxas.metrics import MetricManager
from tqdm import tqdm
import os
import os.path as osp
import json
import matplotlib.pyplot as plt
import torch_geometric.nn as geomnn
from torch_geometric.nn import MessagePassing, global_mean_pool, GATConv, GATv2Conv


class GNNTrainer():
    """
    Class to train and validate GNN models. Modified from https://github.com/AI4-XAS/XASNet-XAI.
    """
    def __init__(
        self, 
        model: Any, 
        model_name: str, 
        device: str,
        metric_path: str
        ):
        """
        Args:
            model (Any): GNN model to train.    
            model_name (str): The name of the model to save. 
            device (str): Device to train on, i.e. cpu or cuda.
            metric_path (str): path to save the metrics data.
        """
        self.model = model
        self.model_name = model_name 

        self.metrics = MetricManager(modes = ['train', 'val','train_every','val_every'])
        # load the previous metrics from last training
        if osp.exists(osp.join(metric_path, "train_metrics.csv")):
            for mode in self.metrics.modes:
                df = pd.read_csv(
                    osp.join(metric_path, f"{mode}_metrics.csv")
                    )
                for col in list(df.columns):
                    self.metrics.outputs[mode][col] = list(df[col])
        else:
            pass
        self.metric_path=metric_path
        self.device = device

    def train_val(
        self, 
        train_loader, 
        val_loader, 
        optimizer,
        loss_fn, 
        scheduler,
        epochs,
        output_step: int = 50,
        start_epoch: int =0,
        save_temp=False
        ):

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        prev_loss = torch.tensor(float('inf'), device=self.device)

        start.record()
        for epoch in tqdm(range(start_epoch+0, start_epoch+epochs), position=0, leave=True):
            train_loss = 0
            num_train = 0
            val_loss = 0
            num_val = 0
            self.model.train()
            for batch in train_loader:

                #correct the batch index for site vectors
                correction_list=[]
                correction=0
                for i in range(0,len(batch.edge_sources)):
                    if i!=0 and batch.edge_sources[i]==0 and batch.edge_sources[i]<batch.edge_sources[i-1]:
                        correction=correction+batch.edge_sources[i-1]+1
                    correction_list.append(correction)
                correction_list=torch.LongTensor(correction_list).to(self.device)

                batch = batch.to(self.device)

                nodes,edge_sources,edge_targets,edge_distance,mask,batch_seg,combine_sets,plane_wave = batch.x,batch.edge_sources,batch.edge_targets,batch.edge_distance,batch.mask,batch.batch,batch.combine_sets,batch.plane_wave

                optimizer.zero_grad()

                pred = self.model(nodes,edge_sources,edge_targets,edge_distance,mask,batch_seg,combine_sets,plane_wave,correction_list)

                loss = loss_fn(
                    pred.view(-1, 1),
                    batch.spectrum_i.view(-1, 1)
                )
                loss.backward()
                train_loss += loss
                num_train += batch.num_graphs
                optimizer.step()

                end.record()
                torch.cuda.synchronize()
                time=f"{start.elapsed_time(end)/6e4:.2f} mins"
                lr=round(float(scheduler.get_lr()[0]), 5)
                self.metrics.store_metrics(
                    mode='train_every',
                    epoch=epoch,
                    time=time,
                    loss=round(float(loss), 5),
                    lr=lr
                    )
            scheduler.step()
            
            avg_train_loss = train_loss / num_train 

            with torch.no_grad():
                self.model.eval()
                for batch in val_loader:

                    correction_list=[]
                    correction=0
                    for i in range(0,len(batch.edge_sources)):
                        if i!=0 and batch.edge_sources[i]==0 and batch.edge_sources[i]<batch.edge_sources[i-1]:
                            correction=correction+batch.edge_sources[i-1]+1
                        correction_list.append(correction)
                    correction_list=torch.LongTensor(correction_list).to(self.device)

                    batch = batch.to(self.device)
                    nodes,edge_sources,edge_targets,edge_distance,mask,batch_seg,combine_sets,plane_wave = batch.x,batch.edge_sources,batch.edge_targets,batch.edge_distance,batch.mask,batch.batch,batch.combine_sets,batch.plane_wave

                    pred = self.model(nodes,edge_sources,edge_targets,edge_distance,mask,batch_seg,combine_sets,plane_wave,correction_list)

                    loss = loss_fn(
                        pred.view(-1, 1), 
                        batch.spectrum_i.view(-1, 1)
                    )
                    val_loss += loss
                    num_val += batch.num_graphs
                    end.record()
                    torch.cuda.synchronize()
                    time=f"{start.elapsed_time(end)/6e4:.2f} mins"
                    lr=round(float(scheduler.get_lr()[0]), 5)
                    self.metrics.store_metrics(
                        mode='val_every',
                        epoch=epoch,
                        time=time,
                        loss=round(float(loss), 5),
                        lr=lr
                        )
            avg_val_loss = val_loss / num_val


            if save_temp:
                self._save_temp_model(epoch=epoch)
            if avg_val_loss < prev_loss:
                self._save_model()
                prev_loss = avg_val_loss
            


            if epoch % int(output_step) == 0:
                end.record()
                torch.cuda.synchronize()
                time=f"{start.elapsed_time(end)/6e4:.2f}"
                lr=round(float(scheduler.get_lr()[0]), 5)
                self.metrics.store_metrics(
                    mode='train',
                    epoch=epoch,
                    time=time,
                    loss=round(float(avg_train_loss), 5),
                    lr=lr
                    )
                self.metrics.store_metrics(
                    mode='val',
                    epoch=epoch,
                    time=time,
                    loss=round(float(avg_val_loss), 5),
                    lr=lr
                    )
                self.save_metrics(path=self.metric_path)
                
                print(f"time = {time} mins")
                print(f"epoch {epoch} | average train loss = {avg_train_loss:.5f}",
                    f" and average validation loss = {avg_val_loss:.5f}",
                    f" |learning rate = {lr:.5f}")


    def predict(
        self, 
        graph, 
        model_path: str
        ):
        if model_path:
            assert model_path.endswith('.pt')
            self.model.load_state_dict(torch.load(model_path))

        graph = graph.to(self.device)
        nodes,edge_sources,edge_targets,edge_distance,mask,combine_sets,plane_wave = graph.x,graph.edge_sources,graph.edge_targets,graph.edge_distance,graph.mask,graph.combine_sets,graph.plane_wave
        batch_seg = torch.tensor(np.repeat(0, nodes.shape[0]), device=self.device,dtype=torch.int64)

        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            graph_pred = self.model(nodes,edge_sources,edge_targets,edge_distance,mask,batch_seg,combine_sets,plane_wave)
            graph_pred = graph_pred.cpu().numpy().flatten()
        return graph_pred

    def save_metrics(self, path: str = './metrics/'):
        if not osp.exists(path):
            os.makedirs(path)
        for mode in self.metrics.modes:
            df = pd.DataFrame(self.metrics.outputs[mode])
            df.to_csv(osp.join(path, f"{mode}_metrics.csv"), index=False)
        

    def _save_model_params(self, path: str = './metrics/'):
        if not osp.exists(path):
            os.mkdir(path)

        params = {}
        for k, v in self.model.__dict__.items():
            if isinstance(v, (str, int, float, list)):
                params[k] = v
        
        with open(osp.join(path, 'params.json'), 'w') as fout:
            json.dump(params, fout)
       
    def _save_model(self):
        if not osp.exists('./best_model'):
            os.mkdir('./best_model')
        path = osp.join('./best_model', self.model_name + '.pt')
        torch.save(self.model.cpu().state_dict(), path)
        self.model.to(self.device)

    def _save_temp_model(self,epoch):
        if not osp.exists('./best_model/temp/'+self.model_name):
            os.makedirs('./best_model/temp/'+self.model_name)
        
        path = osp.join('./best_model/temp/'+self.model_name, self.model_name +'epoch_%i'%epoch+ '.pt')
        torch.save(self.model.cpu().state_dict(), path)
        self.model.to(self.device)    
    

