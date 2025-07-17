# Crystal Graph Neural Network for XANES Prediciton (CGXAS)
## Overview
CGXAS is a crystal graph neural network is designed to predict the X-ray Absorption Near Edge Structure (XANES) spectrum prediction from the given crystal structure of material. CGXAS_Uni model is trained with a universal dataset containing the simulated XANES spectra covering 44 elements. CGXAS_Exp_S model is finetuned from CGXAS_Uni model with a small experimental dataset containing 48 S K edge XANES spectra.

## Usage

### 1. Dataset Construction
```shell
python ./Dataprocess.py
```

### 2. CGXAS_Uni Training
```shell
python ./Train_Val_Uni.py
```

### 3. CGXAS_Exp_S Training
```shell
python ./Transfer_Exp.py
```

### 4. Plot the Predicted Spectrum
See the examples in Plot_test.ipynb

### Notice:
The dataset for CGXAS_Uni training is constructed based on the simulated XANES data in previous work (Mathew, K. et al. High-throughput computational X-ray absorption spectroscopy. Sci. Data 5:180151 doi: 10.1038/sdata.2018.151 (2018).), which is not directly provided in this demo. Its raw data can be downloaded from https://doi.org/10.6084/m9.figshare.c.3946561 and processed with Dataprocess.py. 
