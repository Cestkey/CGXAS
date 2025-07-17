import os.path
import json
import scipy
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.cif import CifWriter
from mp_api.client import MPRester
from pymatgen.io.cif import CifParser

def build_config_atom_num(cif_path,config_path):
    """""
    Encode atom number and save the configuration file.
    Args:
        cif_path (str): the path to the directory of all .cif files.
        config_path (str): the path to save the configuration file.  
    Return:
    config(dict): atomic numbers and their codings.

    """""
    atoms=[]
    all_files = sorted(glob.glob(os.path.join(cif_path,'*.cif')))
    for path in tqdm(all_files):
        crystal = Structure.from_file(path)
        atoms += list(crystal.atomic_numbers)
    unique_z = np.unique(atoms)
    num_z = len(unique_z)
    print('unique_z:', num_z)
    print('min z:', np.min(unique_z))
    print('max z:', np.max(unique_z))
    z_dict = {z:i for i, z in enumerate(unique_z)}
    # Configuration file
    config = dict()
    config["atomic_numbers"] = unique_z.tolist()
    config["node_vectors"] = np.eye(num_z,num_z).tolist() # One-hot encoding
    with open(config_path, 'w') as f:
        json.dump(config, f)
    return config

def build_config_valence(cif_path,config_path):
    """""
    Encode valence and save the configuration file.
    Args:
        cif_path (str): the path to the directory of all .cif files.
        config_path (str): the path to save the configuration file.  
    Return:
    config(dict): valence and their codings.

    """""
    valences=[]
    all_files = sorted(glob.glob(os.path.join(cif_path,'*.cif')))
    for path in tqdm(all_files):
        parser = CifParser(path)
        structure = parser.get_structures()[0]
        for site in structure.sites:
            valences += [site.specie.common_oxidation_states,]
    unique_z = np.unique(valences)
    num_z = len(unique_z)
    print('unique_z:', num_z)
    print('min z:', np.min(unique_z))
    print('max z:', np.max(unique_z))
    z_dict = {z:i for i, z in enumerate(unique_z)}
    # Configuration file
    config = dict()
    config["atomic_numbers"] = unique_z.tolist()
    config["node_vectors"] = np.eye(num_z,num_z).tolist() # One-hot encoding
    with open(config_path, 'w') as f:
        json.dump(config, f)
    return config


def graph_from_raw_data(path, cif_dir, list_path, config_path, cutoff, max_num_nbr, max_graph=10000):
    """""
    Build crystal graphs from .cif file and store them as .npz files.
    Args:
        path (str): the path to store the crystal graphs.
        cif_dir (str): the directory to the .cif files.
        list_path (str): the path to the list of materials to be processed.
        config_path (str): path to store the atom features.
        cutoff (float): the cut off distance when buiding the edge.
        max_num_nbr (int): maximun numbers of neighboring atoms when buiding the edge.
        max_graph (int): maximun numbers of graphs in one .npz file.

    """""    
    if not os.path.exists(path):
        os.makedirs(path)
    graph_list=pd.read_csv(list_path,header=0)
    graph_list=graph_list.values.tolist()

    graphs=dict()
    for i,name in enumerate(graph_list): 
        file=os.path.join(cif_dir, name[0]+'.cif')
        if os.path.exists(file)==False:
            with MPRester("pPUSS0vsKV6Ts17TRgq2PJ5gLkixbOcu") as mpr:
                structure = mpr.get_structure_by_material_id(name[0])
                cif_writer = CifWriter(structure)
                cif_writer.write_file(file)
            if os.path.exists(file)==False:
                print("CIF of "+name[0]+" does not exists.")
                continue
        lattice,atom_fea,nbr_fea_idx,nbr_distance,nbr_subtract,volume, mask = process(config_path,file,cutoff,max_num_nbr,name[1])
        graphs[name[0]+'_'+name[1]] = (lattice,atom_fea,nbr_fea_idx,nbr_distance,nbr_subtract,volume, mask)
        if i%max_graph==0 and i>0:
            print('{} graphs constructed'.format(i))
            np.savez_compressed(os.path.join(path,"MP_graph_data%i.npz"%int(i/max_graph)), graph_dict=graphs)
            graphs=dict()
            
    np.savez_compressed(os.path.join(path,"MP_graph_data%i.npz"%int(i/max_graph+1)), graph_dict=graphs)        
    print('finish constructe the graph')    

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

def process(config_path,data_path,radius,max_num_nbr,site):
    """""
    Derive parameters to build crystal graph .
    Args:
        config_path (str): path to configuration file of atom features.
        data_path (str): the path to the .cif file.  
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
    return lattice,atom_fea,nbr_fea_idx,nbr_distance,nbr_subtract,volume, mask

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
    if min(x)>min(x_new):
        x=np.append(np.linspace(min(x_new),min(x),10),x)
        y=np.append(np.zeros(10),y)
    if max(x)<max(x_new):
        x=np.append(x,np.linspace(max(x),max(x_new),10))
        y=np.append(y,np.ones(10))       
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
