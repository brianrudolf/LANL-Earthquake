""" step #4

This model training script loads in predefined train and validation data splits and trains multiple
models (in this case, there are 4 CV folds available for training). 

The script is configured to concisely print out configured hyper parameters of the model (SVR in this case),
and the metrics that are used to gauge the model's performance. Mean absolute error is used as that is the metric 
used by the Kaggle competition. The SVR score is used to add a second comparable metric.

Both the model and the scaler used are saved to disk so that future predictions can be made using the same conditions. 
The scaler is fitted by the training data (different for each fold), and used to transform both the training data and 
the validation data. Future test predictions will be made after scaling the test data using the scaler fitted on the training data. 

"""


import numpy as np
import pandas as pd
import os, time, pickle
from tqdm import tqdm
from joblib import dump, load

from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

np.set_printoptions(threshold=10_000)
np.random.seed(seed=5)

data_folder = '172_stats_folded'


for fold in np.arange(1,5):
# for fold in (2,3):
	print("Fold ", fold)

	X_train = np.load(f'train/{data_folder}/X_train-fold_{fold}.npy')
	X_train = np.nan_to_num(X_train)
	y_train = np.load(f'train/{data_folder}/y_train-fold_{fold}.npy')

	X_val 	= np.load(f'train/{data_folder}/X_val-fold_{fold}.npy')
	X_val 	= np.nan_to_num(X_val)
	y_val 	= np.load(f'train/{data_folder}/y_val-fold_{fold}.npy')

	y_train = np.reshape(y_train, (y_train.shape[0],))
	y_val 	= np.reshape(y_val, (y_val.shape[0], ))

	### Preprocess the data features 
	scaler = StandardScaler()
	# scaler = Normalizer()
	# scaler = MinMaxScaler(feature_range=(0,1))

	scaler.fit(X_train)

	X_train = scaler.transform(X_train)	
	X_val 	= scaler.transform(X_val)

	var_kernel = 'linear'
	var_gamma = 0.002
	var_C = 50

	print("Kernel: ", var_kernel, "\t Gamma: ", var_gamma, "\t C: ", var_C)

	model = SVR(kernel=var_kernel, gamma=var_gamma, C=var_C)
	model.fit(X_train, y_train)


	tr_prediction 	= model.predict(X_train)
	tr_mae 			= mean_absolute_error(y_train, tr_prediction)
	tr_score 		= model.score(X_train, y_train)
	print("Training shape:   ", X_train.shape, "\tModel mae:  ", np.round(tr_mae,3), "\tModel score:  ", np.round(tr_score, 3))

	val_prediction 	= model.predict(X_val)
	val_mae 		= mean_absolute_error(y_val, val_prediction)
	val_score 		= model.score(X_val, y_val)
	print("Validation shape: ", X_val.shape, "\tModel mae:  ", np.round(val_mae, 3), "\tVal score:  ", np.round(val_score, 3))
	print("")

	### save the model and scaler for future predictions
	dump(model, f'models/model_SVR_{data_folder}_{var_kernel}_{var_gamma}_{var_C}_{np.round(val_mae, 3)}.joblib')
	dump(scaler, f'models/scaler_SVR_{data_folder}_{var_kernel}_{var_gamma}_{var_C}_{np.round(val_mae, 3)}.joblib')