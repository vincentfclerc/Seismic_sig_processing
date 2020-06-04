# -*- coding: utf-8 -*-
"""Functionnal_model_Conv_NN_trainings_on_GPU_short_moving_windows.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1S7Z7xnZB1hH2phk-cdqTw7E1H_A_HB15
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoPickP

Created on Tue Jun 18 16:27:39 2019

@author: Vincent4TOMOSs
"""
#from __future__ import absolute_import, division, print_function

import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow
import random
from copy import deepcopy
from sklearn import model_selection
from sklearn.utils import resample
from scipy.signal import hilbert
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import ensemble
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from scipy.signal import hilbert
from datetime import datetime as dt

 
import scipy.fftpack
#from google.colab import drive
#drive.mount('/content/drive')

#print(tensorflow.__version__)

random.seed( 30 )

raw_dataset = pd.read_csv('C:/Users/vince/Dropbox/tomos_proto_classif/fenetres_equilibrees.txt', sep=" ", header=None)

raw_dataset=raw_dataset.dropna(axis=1, how='any', thresh=None, subset=None, inplace=False)
dataset = raw_dataset.copy()


#%% On sépare aléatoirement le jeu de donnée en jeu d'entrainement et de validation
labels = dataset.pop(0)
X_train, X_test, y_train, y_test = model_selection.train_test_split(dataset.values, -labels.values, test_size = 0.20)
def preTransform(Data):
  max0=np.abs(Data).max(axis=1)
  dataset_norm = Data / max0[:,None]

  #enveloppe
  dataset_fourier_abs=np.abs(hilbert(Data)) 
  max1=np.abs(dataset_fourier_abs).max(axis=1)
  dataset_fourier_abs_norm = dataset_fourier_abs[...,150:301] / max1[:,None]

#imag fft
  dataset_fourier_imag=np.imag(scipy.fftpack.fft(Data))
  max2=np.abs(dataset_fourier_imag).max(axis=1)
  dataset_fourier_imag_norm = dataset_fourier_imag[...,150:301] / max2[:,None]
#real fft
  dataset_fourier_real=np.real(scipy.fftpack.fft(Data))
  max3=np.abs(dataset_fourier_real).max(axis=1)
  dataset_fourier_real_norm = dataset_fourier_real[...,150:301] / max3[:,None]
  return dataset_norm,dataset_fourier_abs_norm,dataset_fourier_imag_norm,dataset_fourier_real_norm

X_train_norm,X_train_fourier_abs_norm,X_train_fourier_imag_norm,X_train_fourier_real_norm=preTransform(X_train)
X_test_norm,X_test_fourier_abs_norm,X_test_fourier_imag_norm,X_test_fourier_real_norm=preTransform(X_test)

#%%
X_train_norm[np.isnan(X_train_norm)] = 0
X_test_norm[np.isnan(X_test_norm)] = 0
X_train_fourier_abs_norm[np.isnan(X_train_fourier_abs_norm)] = 0
X_test_fourier_abs_norm[np.isnan(X_test_fourier_abs_norm)] = 0
X_train_fourier_imag_norm[np.isnan(X_train_fourier_imag_norm)] = 0
X_test_fourier_imag_norm[np.isnan(X_test_fourier_imag_norm)] = 0
X_train_fourier_real_norm[np.isnan(X_train_fourier_real_norm)] = 0
X_test_fourier_real_norm[np.isnan(X_test_fourier_real_norm)] = 0
#%%
print(type(X_train),type(X_test),type(y_train),type(y_test))
print(y_test)
#%%
def create_cnn(length,size_filters, filters=(128, 128, 128), regress=False):

	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (length,1)
	chanDim = -1
 
	# define the model input
	inputs = tensorflow.keras.layers.Input(shape=inputShape)
 
	# loop over the number of filters
	for (i, f) in enumerate(filters):
		# if this is the first CONV layer then set the input
		# appropriately
		if i == 0:
			x = inputs
 
		# CONV => RELU => BN => POOL
		x = tensorflow.keras.layers.Conv1D(f, size_filters, padding="same")(x)
		#x = tensorflow.keras.layers.Activation("relu")(x)
		x = tensorflow.keras.layers.LeakyReLU(alpha=0.1)(x)
		x = tensorflow.keras.layers.BatchNormalization(axis=chanDim)(x)
		x = tensorflow.keras.layers.MaxPooling1D(pool_size=2)(x)
	# flatten the volume, then FC => RELU => BN => DROPOUT
	x = tensorflow.keras.layers.Flatten()(x)
	x = tensorflow.keras.layers.Dense(64)(x)
	x = tensorflow.keras.layers.LeakyReLU(alpha=0.1)(x)
	x = tensorflow.keras.layers.BatchNormalization(axis=chanDim)(x)
	x = tensorflow.keras.layers.Dropout(0.5)(x)

	# apply another FC layer, this one to match the number of nodes
	# coming out of the MLP
	x = tensorflow.keras.layers.Dense(32)(x)
	x = tensorflow.keras.layers.LeakyReLU(alpha=0.1)(x)
	x = tensorflow.keras.layers.Dense(16)(x)
	x = tensorflow.keras.layers.LeakyReLU(alpha=0.1)(x)
	# check to see if the regression node should be added
	if regress:
		x = tensorflow.keras.layers.Dense(1, activation="linear")(x)
 
	# construct the CNN
	model = tensorflow.keras.models.Model(inputs, x)
 
	# return the CNN
	return model

"""### Ici je suis censé décrire le fonctionnement du code
on fera ça plus tard
"""

cnn1 = create_cnn(400,11, regress=False)
cnn2 = create_cnn(200,11, regress=False)
cnn3 = create_cnn(200,11, regress=False)
cnn4 = create_cnn(200,11, regress=False)

# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = tensorflow.keras.layers.concatenate([cnn1.output, cnn2.output, cnn3.output, cnn4.output])

lrelu = lambda x: tensorflow.keras.activations.relu(x, alpha=0.1)


# our final FC layer head will have two dense layers, the final one
# being our regression head
x = tensorflow.keras.layers.Dense(128, activation=lrelu)(combinedInput)
x = tensorflow.keras.layers.Dropout(0.1)(x)
x = tensorflow.keras.layers.BatchNormalization(axis=-1)(x)
x = tensorflow.keras.layers.Dense(64, activation=lrelu)(combinedInput)
x = tensorflow.keras.layers.BatchNormalization(axis=-1)(x)
x = tensorflow.keras.layers.Dense(16, activation=lrelu)(combinedInput)
x = tensorflow.keras.layers.Dense(8, activation=lrelu)(combinedInput)
x = tensorflow.keras.layers.Dense(1, activation=lrelu)(x)
 
# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (the
# predicted price of the house)
model = tensorflow.keras.models.Model(inputs=[cnn1.input, cnn2.input, cnn3.input, cnn4.input], outputs=x)

import keras.backend as K
def norme_perso(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    #loss = (sum((abs(y_true_f - y_pred_f))**1.3))**(1/1.3) # norme l 1.3
    if y_true_f == y_pred_f:
      loss=0
    else:
      loss = sum(np.log(abs(y_true_f - y_pred_f/10)))# log
    return(loss)

# compile the model using mean absolute percentage error as our loss,

opt = tensorflow.keras.optimizers.Adam(lr=0.002, decay=1e-3 / 200)
model.compile(loss='mean_absolute_error',
              optimizer=opt,
              metrics=['mean_absolute_error','mean_squared_error'])
model.summary()

class SaveTraining(tensorflow.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 11 == 0: 
      name_json = 'model_training.json'
      name_weights = 'model_training.h5'

      model_json = model.to_json()
      with open(name_json, "w") as json_file:
          json_file.write(model_json)
      # serialize weights to HDF5
      model.save_weights(name_weights)
      print("Saved model to disk")


# train the model
print("[INFO] training model...")
history=model.fit(
	[np.expand_dims(X_train_norm, axis=2), np.expand_dims(X_train_fourier_abs_norm, axis=2),
  np.expand_dims(X_train_fourier_real_norm, axis=2), np.expand_dims(X_train_fourier_imag_norm, axis=2)],
  y_train,
	validation_split = 0.15,
	epochs=50, batch_size=1500)
  #callbacks=[SaveTraining])

#validation_data=([np.expand_dims(X_test_norm, axis=2), np.expand_dims(X_test_fourier_abs_norm, axis=2),
 #                  np.expand_dims(X_test_fourier_real_norm, axis=2), np.expand_dims(X_test_fourier_imag_norm, axis=2)],
  #                 y_test),

name_json = 'model_' + time.strftime("%Y-%m-%d-%H-%M") +'.json'
name_weights = 'model_' + time.strftime("%Y-%m-%d-%H-%M") +'.h5'

model_json = model.to_json()
with open(name_json, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(name_weights)
print("Saved model to disk")

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error ')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,300])


  
  
plot_history(history)

model_json = model.to_json()
with open("/content/drive/My Drive/Manips_Tomos/NNModels/model_" + dt.now().strftime("%d-%m-%Y-%H-%M")+".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/content/drive/My Drive/Manips_Tomos/NNModels/model_" + dt.now().strftime("%d-%m-%Y-%H-%M")+ ".h5")
print("Saved model to disk")

#model = model_from_yaml(open('modelFilteredPasMal.yaml').read()) 
#%%

test_predictions = model.predict([np.expand_dims(X_test_norm, axis=2), np.expand_dims(X_test_fourier_abs_norm, axis=2),
                                  np.expand_dims(X_test_fourier_real_norm, axis=2), np.expand_dims(X_test_fourier_imag_norm, axis=2)])

#%%

errors = (y_test-test_predictions[...,0])
moyenne=np.mean(errors) 
mediane=np.median(errors) 
ecart_type=np.std(errors) 

plt.figure(1)
plt.hist(errors,1000,range=(-200,200))
plt.xlabel('prediction  error in samples  ')
plt.ylabel('bins  ')
plt.title(f'error(observed-predicted) distribution for auto picks ') 
plt.xlim([-50,50])

#plt.savefig("/content/drive/My Drive/Manips_Tomos/NNModels/results_model_" + dt.now().strftime("%d-%m-%Y-%H-%M")+".png")

plt.figure(2)
plt.hist(y_train,60)
plt.xlabel('pick location in samples ')
plt.ylabel('bins (over 648 test points) ')
plt.title('time distribution for picked arrivals in the test set')

plt.figure(3)
plt.scatter(y_test[1:200,...], test_predictions[1:200,...])
plt.xlabel('True Values ')
plt.ylabel('Predictions ')
plt.ylim([0,300])
plt.xlim([0,300])

print((sum(np.sqrt(errors**2))/648))

