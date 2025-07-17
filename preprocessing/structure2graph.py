
import numpy as np
import pandas as pd

from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element

from scipy.special import jn_zeros,jn,sph_harm
import torch

from torch_geometric.data import Data, Dataset
from torch_geometric.data.collate import collate


def structure2graph(config_path,data_path,radius,max_num_nbr,site, name, cutoff,n_grid_K,n_Gaussian):
    """""
    Derive parameters to build crystal graph .
    Args:
        config_path (str): path to configuration file of atom features.
        data_path (str): the path to the file containing structure information. Supported formats include CIF, POSCAR/CONTCAR, CHGCAR, LOCPOT, vasprun.xml, CSSR, Netcdf and pymatgen's JSON-serialized structures.  
        radius (float): the cut off radius to get the neighboring atoms.
        max_num_nbr (int):  maximun numbers of neighboring atoms.
        site (str): absorption site in XAS.
    Return:
    lattice (array): lattice matrix.
    atom_fea (array): features of all atoms in the crystal.
    nbr_fea_idx (array): features of neighboring atoms of each atom in the crystal.
    nbr_distance (array): distances of atoms to their neighboring atoms.
    nbr_subtract (array): vectors of atoms to their neighboring atoms.
    volume (float): volume of the cell.
    mask (array): mask for the absorbing elements.

    """""    
    crystal = Structure.from_file(data_path)
    volume=crystal.lattice.volume
    coords=crystal.cart_coords
    lattice=crystal.lattice.matrix
    atoms=crystal.atomic_numbers
    atom_fea = np.vstack([onehot_atom_features(atom_num=atoms[i], features=["Z","row","nsvalence","npvalence","ndvalence","nfvalence"],config_path=config_path) for i in range(len(crystal))])

    mask = np.array([str(Element.from_Z(atoms[i]))==site for i in range(len(crystal))])

    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea = [], []

    for i,nbr in enumerate(all_nbrs):
        if len(nbr) < max_num_nbr:
            nbr_fea_idx.append(list(map(lambda x: x[2].tolist(), nbr)) +
                                [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append(list(map(lambda x: x[0].coords.tolist(), nbr)) +
                   [[coords[i][0]+radius,coords[i][1],coords[i][2]]] * (max_num_nbr -len(nbr)))
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2].tolist(),
                                        nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[0].coords.tolist(),
                                    nbr[:max_num_nbr])))
    atom_fea=atom_fea.tolist()

    nbr_subtract=[]
    nbr_distance=[]

    for i in range(len(nbr_fea)):
        if nbr_fea[i] != []:
            x=nbr_fea[i]-coords[:,np.newaxis,:][i]
            nbr_subtract.append(x)
            nbr_distance.append(np.linalg.norm(x, axis=1).tolist())
        else:
            nbr_subtract.append(np.array([]))
            nbr_distance.append(np.array([]))

    nbr_fea_idx = np.array(nbr_fea_idx) 


    #lattice, nodes, nei,distance,vector,volume,mask=lattice,atom_fea,nbr_fea_idx,nbr_distance,nbr_subtract,volume, mask

    n_nodes = len(atom_fea) 
            
    nodes = np.array(atom_fea, dtype=np.float32)
    edge_sources = np.concatenate([[i] * len(nbr_fea_idx[i]) for i in range(n_nodes)])
    edge_targets=np.concatenate(nbr_fea_idx)
    edge_vector = np.array(nbr_subtract, dtype=np.float32)
    edge_index = np.concatenate([range(len(nbr_fea_idx[i])) for i in range(n_nodes)])
    vectorij= edge_vector[edge_sources,edge_index]
    edge_distance = np.array(nbr_distance, dtype=np.float32)
    distance= edge_distance[edge_sources,edge_index]
    combine_sets=[]
    # gaussian radial
    N=n_Gaussian
    for n in range(1,N+1):
        phi=Phi(distance,cutoff)
        G=gaussian(distance,miuk(n,N,cutoff),betak(N,cutoff))
        combine_sets.append(phi*G)
    combine_sets=np.array(combine_sets, dtype=np.float32).transpose()

     # plane wave
    grid=n_grid_K
    kr=np.dot(vectorij,get_Kpoints_random(grid,lattice,volume).transpose()) 
    plane_wave=np.cos(kr)/np.sqrt(volume)  

    nodes = torch.Tensor(nodes)
    edge_distance = torch.Tensor(edge_distance)
    edge_sources = torch.LongTensor(edge_sources)
    edge_targets = torch.LongTensor(edge_targets)
    edge_index=torch.Tensor(edge_index)
    combine_sets=torch.Tensor(combine_sets)
    plane_wave=torch.Tensor(plane_wave)
    spectrum=torch.Tensor(np.zeros(100))
    energy=torch.Tensor(np.zeros(100))
    mask=torch.Tensor(mask)
    data = Data(x=nodes, edge_index=edge_index,edge_sources=edge_sources,edge_targets=edge_targets,edge_distance=edge_distance, combine_sets=combine_sets, plane_wave=plane_wave, mask=mask, spectrum=spectrum, energy=energy, name=name)

    return data

def onehot_atom_features(atom_num, features,config_path):
    """""
    Derive parameters to build crystal graph .
    Args:
        atom_num (int): atomic number of element.
        features (list): features to be encoded.  
        config_path (str): path to store the atom features.
    Return:
    one_hot_code (array): one hot code for this element.
    """""   
    config=pd.read_csv(config_path,header=0)
    one_hot_list=[]
    for j in features:
        if j=="Z":
            temp=np.zeros(103)
            temp[atom_num-1]=1
            one_hot_list.append(temp)
        elif j=="row":
            temp=np.zeros(7)
            temp[config[j][atom_num-1]-1]=1     
            one_hot_list.append(temp)  
        elif j== "nsvalence":  
            temp=np.zeros(3)
            temp[config[j][atom_num-1]]=1     
            one_hot_list.append(temp)  
        elif j== "npvalence":  
            temp=np.zeros(7)
            temp[config[j][atom_num-1]]=1     
            one_hot_list.append(temp) 
        elif j== "ndvalence":  
            temp=np.zeros(11)
            temp[config[j][atom_num-1]]=1     
            one_hot_list.append(temp)  
        elif j== "nfvalence":  
            temp=np.zeros(15)
            temp[config[j][atom_num-1]]=1     
            one_hot_list.append(temp)                  
    one_hot_code=np.hstack(one_hot_list)
    return one_hot_code


def a_SBF(alpha,l,n,d,cutoff):
    root=float(jn_zeros(l,n)[n-1])
    return jn(l,root*d/cutoff)*sph_harm(0,l,np.array(alpha),0).real*np.sqrt(2/cutoff**3/jn(l+1,root)**2)

def a_RBF(n,d,cutoff):
    return np.sqrt(2/cutoff)*np.sin(n*np.pi*d/cutoff)/d

def get_Kpoints_random(q,lattice,volume):
    a0=lattice[0,:]
    a1=lattice[1,:]
    a2=lattice[2,:]
    unit=2*np.pi*np.vstack((np.cross(a1,a2),np.cross(a2,a0),np.cross(a0,a1)))/volume
    ur=[(2*r-q-1)/2/q for r in range(1,q+1)]
    points=[]
    for i in ur:
        for j in ur:
            for k in ur:
                points.append(unit[0,:]*i+unit[1,:]*j+unit[2,:]*k)
    points=np.array(points) 
    return points  


def Phi(r,cutoff):
    return 1-6*(r/cutoff)**5+15*(r/cutoff)**4-10*(r/cutoff)**3
def gaussian(r,miuk,betak):
    return np.exp(-betak*(np.exp(-r)-miuk)**2)
def miuk(n,K,cutoff):
    # n=[1,K]
    return np.exp(-cutoff)+(1-np.exp(-cutoff))/K*n
def betak(K,cutoff):
    return (2/K*(1-np.exp(-cutoff)))**(-2)