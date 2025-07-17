import sys
import torch
import time
import json
import os
import copy
import numpy as np
import pandas as pd
import math
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from torch_geometric.nn import  global_mean_pool
import torch.nn as nn
from torch.nn import ( Linear, Bilinear, Sigmoid, Softplus, ELU, ReLU, SELU,
                       CELU, BatchNorm1d, ModuleList, Sequential,Tanh,Softmax,SiLU)
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
import re

def get_activation(name):
    act_name = name.lower()
    m = re.match(r"(\w+)\((\d+\.\d+)\)", act_name)
    if m is not None:
        act_name, alpha = m.groups()
        alpha = float(alpha)
        print(act_name, alpha)
    else:
        alpha = 1.0
    if act_name == 'softplus':
        return Softplus()
    elif act_name == 'softmax':
        return Softmax(dim=1)
    elif act_name == 'elu':
        return ELU(alpha)
    elif act_name == 'relu':
        return ReLU()
    elif act_name == 'selu':
        return SELU()
    elif act_name == 'celu':
        return CELU(alpha)
    elif act_name == 'sigmoid':
        return Sigmoid()
    elif act_name == 'tanh':
        return Tanh()
    elif act_name == 'silu':
        return SiLU()    
    else:
        raise NameError("Not supported activation: {}".format(name))


def _bn_act(num_features, activation, use_batch_norm=False):
    # batch normal + activation
    if use_batch_norm:
        if activation is None:
            return BatchNorm1d(num_features)
        else:
            return Sequential(BatchNorm1d(num_features), activation)
    else:
        return activation

class NodeEmbedding(Module):
    """
    Node Embedding layer
    """
    def __init__(self, in_features, out_features, activation=Sigmoid(),
                 use_batch_norm=False, bias=False):
        super(NodeEmbedding, self).__init__()
        self.linear = Linear(in_features, out_features, bias=bias)
        self.activation = _bn_act(out_features, activation, use_batch_norm)

    def forward(self, input):
        output=self.linear(input)
        output = self.activation(output)
        return output


class OLP(Module):
    """
    One Layer Perceptron
    """    
    def __init__(self, in_features, out_features, activation=ELU(),
                use_batch_norm=False, bias=False):
        super(OLP, self).__init__()
        self.linear = Linear(in_features, out_features, bias=bias)
        self.activation = _bn_act(out_features, activation, use_batch_norm)

    def forward(self,  input):
        z = self.linear(input)
        if self.activation:
            z = self.activation(z)
        return z


class Mask_pooling(Module):
    """
    Mask Pooling Layer 
    """ 
    def __init__(self):
        super(Mask_pooling, self).__init__()

    def forward(self, input,mask,batch_seg):

        node_list=[]
        for i in range(0,len(mask)):
            if mask[i]!=0:
                node_list.append(i)
        output = global_mean_pool(input[node_list,:], batch_seg[node_list])
        return output
    
class Vector_normalization(Module):
    def __init__(self):
        super(Vector_normalization, self).__init__()

    def forward(self, input):
        if len(input.shape)==1:
            input=input.reshape(1,-1)
        output=input/torch.sqrt(torch.sum(input*input,dim=1).reshape(-1,1))
        return output    

class GatedGraphConvolution(Module):
    """
    Gated Graph Convolution Layer of geo-CGNN
    modification from https://github.com/Tinystormjojo/geo-CGNN
    """
    def __init__(self, in_features, out_features,n_grid_K,n_Gaussian, gate_activation=Sigmoid(), use_edge_batch_norm=False,
                 bias=False, MLP_activation=ELU()):
        super(GatedGraphConvolution, self).__init__()
        k1= n_Gaussian # k is the number of basis
        k2=n_grid_K**3
        self.linear1_vector = Linear(k1, out_features, bias=bias) # linear for combine sets
        self.linear1_vector_gate = Linear(k1, out_features, bias=bias) # linear for combine sets
        self.activation1_vector_gate = _bn_act(out_features, gate_activation, use_edge_batch_norm)
        self.linear2_vector = Linear(k2, out_features, bias=bias) # linear for plane waves
        self.linear2_vector_gate = Linear(k2, k2, bias=bias) # linear for plane waves
        self.activation2_vector_gate = _bn_act(k2, gate_activation, use_edge_batch_norm)

        self.linear_gate = Linear(in_features, out_features, bias=bias)
        self.activation_gate = _bn_act(out_features, gate_activation, use_edge_batch_norm)

        self.linear_MLP = Linear(in_features, out_features, bias=bias)
        self.activation_MLP = _bn_act(out_features, MLP_activation, use_edge_batch_norm)

   
    def forward(self, input, edge_sources, edge_targets, rij ,combine_sets,plane_wave,cutoff,batch_correction):
        if batch_correction!=None:
            ni = input[edge_sources+batch_correction].contiguous()
            nj = input[edge_targets+batch_correction].contiguous()
        else:
            ni = input[edge_sources].contiguous()
            nj = input[edge_targets].contiguous()            
        rij=rij.unsqueeze(1).contiguous().view(-1,1)
        mask=rij<cutoff
        delta= (ni-nj)/rij
        final_fe=torch.cat([ni,nj,delta],dim=1)
        del ni,nj,delta
        torch.cuda.empty_cache()
 
        e_gate = self.activation_gate(self.linear_gate(final_fe))
        e_MLP = self.activation_MLP(self.linear_MLP(final_fe))

        z1 = self.linear1_vector(combine_sets)
        gate=self.activation2_vector_gate(self.linear2_vector_gate(plane_wave))
        z2 = self.linear2_vector(plane_wave*gate)
        z =  e_gate * e_MLP * (z1+z2) * mask
        del z1,z2,e_gate,e_MLP
        torch.cuda.empty_cache()
        output = input+torch.sum(z.view(input.shape[0],12,input.shape[1]),dim=1)
        return output

class GraphConvolution(Module):
    """
    Gated Graph Convolution Layer of CGXAS
    """
    def __init__(self, in_features, hidden_features, out_features,n_grid_K,n_Gaussian, gate_activation=Sigmoid(), MLP_activation=ELU(),use_edge_batch_norm=False,use_node_batch_norm=False,
                 bias=False):
        super(GraphConvolution, self).__init__()    
        self.conv=GatedGraphConvolution(in_features, hidden_features, n_grid_K,n_Gaussian,
                gate_activation=gate_activation, 
                MLP_activation=MLP_activation, 
                use_edge_batch_norm=use_edge_batch_norm,
                bias=bias)
        self.norm=Vector_normalization()
        self.olp=OLP(hidden_features, out_features, activation=MLP_activation, use_batch_norm=use_node_batch_norm, bias=bias)
        
    def forward(self, input, edge_sources, edge_targets, rij ,combine_sets,plane_wave,cutoff,batch_correction):
        x=self.conv(input, edge_sources, edge_targets, rij ,combine_sets,plane_wave,cutoff,batch_correction)
        x=self.norm(x)
        x=self.olp(x)
        return x
    

class CGXAS(nn.Module):
    def __init__(self,n_node_feat, n_hidden_feat, out_feat, conv_bias, n_GNN, n_MLP, node_activation, MLP_activation, use_node_batch_norm, use_edge_batch_norm, cutoff, n_grid_K, n_Gaussian):
        """
        Args:
            n_node_feat (int): length of node vector in input graph. 
            n_hidden_feat (int): length of hidden layer vector.
            out_feat (int): length of output vector.
            conv_bias (bool): whether bias is used in convolution layer.
            n_GNN (int): number of GNN layers.
            n_MLP (int): number of MLP layers.
            node_activation (str): activation function of convolution gate in processing node features.
            MLP_activation (str): activation function of MLP layer.
            use_node_batch_norm (bool): whether batch norm is used in processing node features.
            use_edge_batch_norm (bool): whether batch norm is used in processing edge features.
            cutoff (float): cutoff radius of edge.
            n_grid_K (int): number of k space grids.
            n_Gaussian (int): number of gaussian sets.
        """

        super(CGXAS, self).__init__()
        self.cutoff=cutoff
        self.n_GNN=n_GNN
        self.n_MLP=n_MLP
        node_activation=get_activation(node_activation)
        MLP_activation=get_activation(MLP_activation)
        self.embedding = NodeEmbedding(n_node_feat, n_hidden_feat)
        n2v_concatent_feat = n_hidden_feat*3 #ni+nj+delta

        self.GNN = [GraphConvolution(n2v_concatent_feat, n_hidden_feat, n_hidden_feat, n_grid_K,n_Gaussian,
                gate_activation=node_activation, 
                MLP_activation=MLP_activation, 
                use_edge_batch_norm=use_edge_batch_norm,
                use_node_batch_norm=use_node_batch_norm,
                bias=conv_bias) for _ in range(n_GNN)]

        self.GNN=ModuleList(self.GNN)

        self.dropout = torch.nn.Dropout(p=0.3)      

        # final linear regression
        self.MLP=[OLP(int(n_hidden_feat), int(n_hidden_feat) , activation=MLP_activation, use_batch_norm=use_node_batch_norm, bias=conv_bias) for _ in range(1,n_MLP)]
        self.MLP+=[OLP(int(n_hidden_feat), int(out_feat) , activation=MLP_activation, use_batch_norm=use_node_batch_norm, bias=conv_bias),]
        self.MLP=ModuleList(self.MLP)
        self.relu=ReLU()
        # pooling for final results
        self.pooling=Mask_pooling()        

    
    def forward(self,nodes,edge_sources,edge_targets,edge_distance,mask,batch_seg,combine_sets,plane_wave,batch_correction=None,output_sv=False):
        x = self.embedding(nodes) 
        layer_out=[]
        layer_out.append(x)

        for i in range(self.n_GNN):                
            x = self.GNN[i](x, edge_sources, edge_targets, edge_distance, combine_sets, plane_wave, self.cutoff, batch_correction)
            layer_out.append(x/math.factorial(i+1))
 
        site_vec=torch.sum(torch.stack(layer_out, dim=0),dim=0)
        del layer_out
        y=site_vec
        y=self.dropout(y)
        for i in range(self.n_MLP): 
            y=self.MLP[i](y)
            
        y=self.relu(y)
        y=self.pooling(y,mask,batch_seg)

        if output_sv:
            return y.squeeze(),site_vec
        else:
            return y.squeeze()
