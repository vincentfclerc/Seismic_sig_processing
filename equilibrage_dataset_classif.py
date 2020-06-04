# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:25:08 2020

@author: vince
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
from sklearn.utils import resample

#%%

raw_dataset_X = pd.read_csv('C:/Users/vince/Dropbox/tomos_proto_classif/toutes_fenetres_X.txt', sep=" ", header=None)
raw_dataset_Y = pd.read_csv('C:/Users/vince/Dropbox/tomos_proto_classif/toutes_fenetres_Y.txt', sep=" ", header=None)
raw_dataset_Z = pd.read_csv('C:/Users/vince/Dropbox/tomos_proto_classif/toutes_fenetres_Z.txt', sep=" ", header=None)

#%%
sig_win=np.concatenate((raw_dataset_X, raw_dataset_Y, raw_dataset_Z))
#%%
test_sum=np.sum(sig_win,axis=1)

sig_win = sig_win[test_sum!=0,:]
#%%

TOL=20
picks_P = sig_win[...,0]
picks_S = sig_win[...,1]
arrivee_P = copy.deepcopy(picks_P)
arrivee_S = copy.deepcopy(picks_S)
arrivee_P[(arrivee_P+TOL) > 0] = 1
arrivee_P[arrivee_P < 0] = 0

arrivee_S[(arrivee_S+TOL) > 0] = 1
arrivee_S[arrivee_S < 0] = 0

labels_P=arrivee_P
labels_S=arrivee_S

sig_win = np.delete(sig_win ,0, axis=1)
sig_win[:, 0] =  labels_P
        
dataset_majority = sig_win[arrivee_P==1]
dataset_minority = sig_win[arrivee_P==0]
df_majority=pd.DataFrame(dataset_majority)
df_minority=pd.DataFrame(dataset_minority)
# downsample majority class
dataset_majority_downsampled = resample(df_majority, 
                         replace=True,     
                         n_samples=len(dataset_minority),    # to match majority class
                         random_state=124) # reproducible results
dataset_sampled = np.concatenate([dataset_majority_downsampled, dataset_minority])

np.savetxt(f'C:/Users/vince/Dropbox/tomos_proto_classif/fenetres_equilibrees_classif.txt', dataset_sampled, fmt='%1.1f')

