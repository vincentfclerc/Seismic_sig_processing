# -*- coding: utf-8 -*
"""
Created on Fri Feb 21 11:27:16 2020

@author: vince


ici l'idée est de découper des fenetres dans les trois composantes et de sauver
 un fichier par fenetre
ça fait plein de fichiers'
"""


import pandas as pd
import numpy as np
from sklearn.utils import resample
import copy

# %%


for file in range(1, 2): #rajouter un zero dans le nom du fichier si <10
    
    print(file)
    raw_dataset = pd. read_csv(f'C:/Users/vince/Dropbox/tomos_proto_classif/data_60sec_PS_3D_sigPS_001.txt', sep=" ", header=None)

    raw_dataset = raw_dataset.dropna(axis=1, how='any', thresh=None,
                                     subset=None, inplace=False)
    dataset = raw_dataset.to_numpy()

    copy_raw_dataset = copy.deepcopy(dataset)

    TOL = 50

    
    picks_P = copy_raw_dataset[...,1]
    picks_S = copy_raw_dataset[...,2]
    
    
    LENGTH_SIGNAL = 300
    #array_starts_windows_long = np.arange(0, 6001, LENGTH_SIGNAL)
    array_starts_windows_long = np.round(np.linspace(0, 5700, 58))
    array_starts_windows = array_starts_windows_long[:-1].copy()
    matrix_dataset = copy_raw_dataset
    sig_win_X = np.zeros(shape=(58*2000, (LENGTH_SIGNAL)+2))
    sig_win_Y = np.zeros(shape=(58*2000, (LENGTH_SIGNAL)+2))
    sig_win_Z = np.zeros(shape=(58*2000, (LENGTH_SIGNAL)+2))
    c_it = 0
    c_it_empty = 0
    for trace in range(1000):
        print(trace)
        index=trace+1
        pick_P = picks_P[trace]
        pick_S = picks_S[trace]
    
        for startf in array_starts_windows:
            start=int(startf)
            sig_win_X[c_it][2:(LENGTH_SIGNAL+2)] = matrix_dataset[trace][start+4:(start+4+LENGTH_SIGNAL)]
            sig_win_Y[c_it][2:(LENGTH_SIGNAL+2)] = matrix_dataset[trace][start+6006:(start+LENGTH_SIGNAL+6006)]
            sig_win_Z[c_it][2:(LENGTH_SIGNAL+2)] = matrix_dataset[trace][start+12008:(start+LENGTH_SIGNAL+12008)]
    
            sig_win_X[c_it][0] = pick_P-start
            sig_win_X[c_it][1] = pick_S-start
            sig_win_Y[c_it][0] = pick_P-start
            sig_win_Y[c_it][1] = pick_S-start
            sig_win_Z[c_it][0] = pick_P-start
            sig_win_Z[c_it][1] = pick_S-start
            c_it += 1
            
            
            #%%
np.savetxt(f'C:/Users/vince/Dropbox/tomos_proto_classif/toutes_fenetres_X.txt',sig_win_X,fmt='%1.1f')
np.savetxt(f'C:/Users/vince/Dropbox/tomos_proto_classif/toutes_fenetres_Y.txt',sig_win_Y,fmt='%1.1f')
np.savetxt(f'C:/Users/vince/Dropbox/tomos_proto_classif/toutes_fenetres_Z.txt',sig_win_Z,fmt='%1.1f')
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

        
dataset_majority = sig_win[arrivee_P==1]
dataset_minority = sig_win[arrivee_P==0]
df_majority=pd.DataFrame(dataset_majority)
df_minority=pd.DataFrame(dataset_minority)
# downsample majority class
dataset_majority_downsampled = resample(df_majority, 
                         replace=False,     
                         n_samples=len(dataset_minority),    # to match majority class
                         random_state=124) # reproducible results
dataset_sampled = np.concatenate([dataset_majority_downsampled, dataset_minority])

np.savetxt(f'C:/Users/vince/Dropbox/tomos_proto_classif/balanced_sliding_windows_file_{file}.txt', dataset_sampled, fmt='%1.1f')

