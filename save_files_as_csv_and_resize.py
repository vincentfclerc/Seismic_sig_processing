# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 11:25:37 2020

@author: vince

Ce fichier récupère les txt de bertrand, les redimensionne et les sauve en csv
"""


import pandas as pd


#%%


for file in range(1, 10):
    print(file)
    raw_dataset = pd.read_csv(f'C:/Users/vince/Dropbox/tomos_proto_classif/data_60sec_PS_3D_sigPS_00{file}.txt',
                              sep=" ", header=None)

    new_dataset = raw_dataset.iloc[:, :18012]
    new_dataset.to_csv(f'c:/Users/vince/Dropbox/tomos_proto_classif/data_60sec_PS_3D_sigPS_00{file}_resized.csv',
                       header=None, index=None, sep=' ', mode='a')
