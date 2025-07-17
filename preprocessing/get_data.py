from mp_api.client import MPRester
from emmet.core.summary import HasProps
from pymatgen.io.cif import CifWriter
import pandas as pd
from emmet.core.xas import Edge, XASDoc, Type
import numpy as np
import os


def get_data(api_key,mpids,path=''):
    """""
    Dowload the .cif files and XANES spectrums from Materials Project.
    Args:
        api_key (str): api key for Material Project.
        mpids (pdframe): list of materials to be downloaded, where the column 'ID' stores the IDs of material in Material Project.  
        path (str): path to save the file.
    """""
    if os.path.exists(os.path.join(path,'spectrum'))==False:
        os.makedirs(os.path.join(path,'spectrum'))
    if os.path.exists(os.path.join(path,'CIF'))==False:
        os.makedirs(os.path.join(path,'CIF'))    

    with MPRester(api_key) as mpr:
        for i in range(0,len(mpids)):     
            structure = mpr.get_structure_by_material_id(mpids['ID'][i])
            xas = mpr.materials.xas.search(material_ids = [mpids['ID'][i],],edge = Edge.K,spectrum_type=Type.XANES)

            for j in range(0,len(xas)):
            
                txt_file = open(os.path.join(path,"absorption_site.txt"), "a", encoding="utf-8")  
                txt_file.write(mpids['ID'][i]+'\t'+str(xas[j].absorbing_element))
                txt_file.close()
                spectrum=np.hstack((xas[j].spectrum.x.reshape(-1,1),xas[j].spectrum.y.reshape(-1,1)))
                np.savetxt(os.path.join(path,'spectrum/'+mpids['ID'][i]+'_'+str(xas[j].absorbing_element)+".csv"), spectrum, delimiter="," )

            cif_writer = CifWriter(structure)
            cif_writer.write_file(os.path.join(path,'CIF/'+mpids['ID'][i]+'.cif'))