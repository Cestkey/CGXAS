import numpy as np
import pandas as pd
import os
import json
import math
from preprocessing.graph import structure2graph
from preprocessing.data_utils import AtomGraphDataset
import torch

def interpolate_curve(x,y,x_new,type):
    """
    Interpolate curve  with specific range.
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
        yl=np.ones(10)*np.mean(y[-5:])
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

def data_split(ndata,ntrain,nval,ntest,seed):
    random_state = np.random.RandomState(seed=seed)
    all_idx = np.arange(ndata)
    all_idx = random_state.permutation(all_idx)
    idxs = {'train' : all_idx[:ntrain],'val' : all_idx[ntrain : ntrain+nval],'test' : all_idx[ntrain+nval:ntrain+nval+ntest]}
    return idxs  

def pack_split(ndata,ntrain,nval,ntest,seed,packsize=1):
    random_state = np.random.RandomState(seed=seed)
    npack=math.ceil(float(ndata)/packsize)
    pack_idx=random_state.permutation(np.arange(npack))
    all_idx = []
    for i in pack_idx:
        all_idx.append(np.arange(i*packsize,min((i+1)*packsize,ndata)))
    all_idx = np.hstack(all_idx)
    idxs = {'train' : all_idx[:ntrain],'val' : all_idx[ntrain : ntrain+nval],'test' : all_idx[ntrain+nval:ntrain+nval+ntest]}
    return idxs  

def spectrum_error(spectrum):
    spectrum.reshape(-1)
    if spectrum[0]>0.5:
        return True
    if max(spectrum) < 1:
        return True
    if min(spectrum) < -0.05:
        return True                
    star=np.where(spectrum>0.9)[0][0]
    if star>=80:
        return True
    if np.min(spectrum[star:])<0.3:
        return True
    return False  


site='S'
path_to_spectrum='./rawdata/spectrum'
path_to_processed_spectrum='./rawdata/spectrum_processed'
path_to_structure_file='./rawdata/structure'
path_to_dataset='./processed_data'


if __name__ == '__main__':
    absorption_site=pd.read_csv('./rawdata/absorption_site/%s.csv'%site,header=0)
    with open('max_Energy1.json') as f:
        max_Energy1 = json.load(f)
    with open('min_Energy1.json') as f:
        min_Energy1 = json.load(f)

    error_index=[]

    for i in range(0,len(absorption_site)):
        try:
            spectrum=pd.read_csv(os.path.join(path_to_spectrum,absorption_site['index'][i]+'-'+str(absorption_site['abs_atom'][i])+'-'+absorption_site['type'][i]+'-'+absorption_site['edge'][i]+'-'+absorption_site['element'][i]+'.csv'),header=None)
        except:
            print(absorption_site['index'][i]+'-'+str(absorption_site['abs_atom'][i])+'-'+absorption_site['type'][i]+'-'+absorption_site['edge'][i]+'-'+absorption_site['element'][i]+' is error')
            continue
        x=spectrum[0].to_numpy()
        y=spectrum[3].to_numpy()
        x_new=np.linspace(min_Energy1[site], max_Energy1[site], 100)     
        try:
            y_new=interpolate_curve(x,y,x_new,'linear')
            #y_new=simpleSmooth(e=x, xanes=y_new, sigma=0.2, kernel='Gauss')
        except:
            error_index.append(absorption_site['index'][i]+'-'+str(absorption_site['abs_atom'][i])+'-'+absorption_site['type'][i]+'-'+absorption_site['edge'][i]+'-'+absorption_site['element'][i])
            continue
        x_new=x_new.reshape(-1,1)
        y_new=y_new.reshape(-1,1)
        if np.isnan(np.sum(y_new)):
            print(absorption_site['index'][i]+'-'+str(absorption_site['abs_atom'][i])+'-'+absorption_site['type'][i]+'-'+absorption_site['edge'][i]+'-'+absorption_site['element'][i]+' is nan')
        spectrum_new=np.hstack((x_new,y_new))
        np.savetxt(os.path.join(path_to_processed_spectrum,absorption_site['index'][i]+'-'+str(absorption_site['abs_atom'][i])+'-'+absorption_site['type'][i]+'-'+absorption_site['edge'][i]+'-'+absorption_site['element'][i]+'.csv'), spectrum_new, delimiter=",")

    root = os.path.join(path_to_dataset, 'Site_XAS_%s.pt'%site)
    CGXASdataset = AtomGraphDataset(root=root)
    absorption_site=pd.read_csv('rawdata/absorption_site/%s.csv'%site,header=0)
    for i in range(0,len(absorption_site)):
        spectrum_file=os.path.join(path_to_processed_spectrum,absorption_site['index'][i]+'-'+str(absorption_site['abs_atom'][i])+'-'+absorption_site['type'][i]+'-'+absorption_site['edge'][i]+'-'+absorption_site['element'][i]+'.csv')
        if os.path.exists(spectrum_file)==False:
            continue
        spectrum=pd.read_csv(spectrum_file,header=None)
        spectrum=spectrum.values

        if spectrum_error(spectrum[:,1]):
            continue

        name=absorption_site['index'][i]+'-'+str(absorption_site['abs_atom'][i])+'-'+absorption_site['type'][i]+'-'+absorption_site['edge'][i]+'-'+absorption_site['element'][i]

        try:
            graph=structure2graph(config_path='config.csv', cif_path=os.path.join(path_to_structure_file,'%s.json'%name), cutoff_r=8, max_num_nbr=12, element=site, sites_index=[absorption_site['abs_atom'][i],], name=name, n_grid_K=4, n_Gaussian=64, spectrum_e=spectrum[:,0], spectrum_i=spectrum[:,1])
        except:
            continue
        CGXASdataset.graph_data.append(graph)

    # save the dataset if it doesn't exists
    if not os.path.exists(root):
        torch.save(CGXASdataset, root)

    idxs=data_split(ndata=len(CGXASdataset.graph_data),ntrain=int(0.6*len(CGXASdataset.graph_data)),nval=int(0.2*len(CGXASdataset.graph_data)),ntest=int(0.2*len(CGXASdataset.graph_data)),seed=64)
    train_data = [CGXASdataset.graph_data[i] for i in idxs['train']]
    val_data = [CGXASdataset.graph_data[i] for i in idxs['val']]
    test_data = [CGXASdataset.graph_data[i] for i in idxs['test']]
    torch.save(train_data, os.path.join(path_to_dataset,'Site_XAS_%s_train.pt'%site))
    torch.save(val_data, os.path.join(path_to_dataset,'Site_XAS_%s_val.pt'%site))
    torch.save(test_data, os.path.join(path_to_dataset,'Site_XAS_%s_test.pt'%site))
