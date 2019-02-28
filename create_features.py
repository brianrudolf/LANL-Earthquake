""" step #3

This feature creation script loads a series of data samples from each pre-defined earthquake period and creates 
features to be used by a learnable model for regression prediction. 

This version creates 4 cross validation folds such that out of 16 earthquake periods, 12 are used for training while 
the remaing 4 are used for cross validation. This strategy prepares these folds such that 4 models can be trained and the
performance on the train and validation data can be compared to select the best model. Each of the 16 periods is used within 
the validation set once.

"""

import numpy as np
import os, time, pickle, random
from tqdm import tqdm

from sklearn import utils

from quake_features import feature_extraction_87, feature_extraction_172, fft_features

np.set_printoptions(threshold=10_000)
np.random.seed(seed=5)


data_folder = '172_stats_folded'

### 4 folds, 12-4 split
pool = np.arange(16)
np.random.shuffle(pool)
val1 = pool[:4]
val2 = pool[4:8]
val3 = pool[8:12]
val4 = pool[12:16]

val_pool = [val1, val2, val3, val4]
loop_time = time.time()

X = []
y = []
for i in range(16):
	### Read data and create features for input samples (train and val)
	print("Processing quake ", i)
	x_tmp = np.load(f'train/data_samples/interval_{i}_X.npy')
	stats = feature_extraction_172(x_tmp)
	ffts  = fft_features(x_tmp)
	X.append(np.concatenate((stats, ffts), axis=1))	
# 	X.append(feature_extraction_172(x_tmp))

	y_tmp = np.load(f'train/data_samples/interval_{i}_y.npy')
	y.append(np.reshape(y_tmp, (y_tmp.shape[0], 1)))
end = time.time()
print("Extracted all features")


for v_i, val in enumerate(val_pool, 1):
	start = time.time()
	print("Creating data based on validation set ", v_i)

	train_pool = list(filter(lambda x: x not in val, pool))

	X_train = [X[x] for x in train_pool]
	y_train	= [y[x] for x in train_pool]

	X_val	= [X[x] for x in val]
	y_val	= [y[x] for x in val]


	X_train = np.vstack(X_train)
	y_train = np.vstack(y_train)

	X_val	= np.vstack(X_val)
	y_val	= np.vstack(y_val)



	print("Train data shape:\t", X_train.shape)
	print("Train label shape:\t", y_train.shape)	

	print("Val data shape: \t", X_val.shape)
	print("Val label shape:\t", y_val.shape)
	
	print("Separated into train and val splits")


	X_train, y_train 	= utils.shuffle(X_train, y_train, random_state=5)
	X_val, y_val 		= utils.shuffle(X_val, y_val, random_state=5)

	### save train and validation data without preprocessing (scaling / standardizing)

	np.save(f'train/{data_folder}/X_train-fold_{v_i}.npy', X_train)
	np.save(f'train/{data_folder}/y_train-fold_{v_i}.npy', y_train)
	np.save(f'train/{data_folder}/X_val-fold_{v_i}.npy', X_val)
	np.save(f'train/{data_folder}/y_val-fold_{v_i}.npy', y_val)
	
	print("Saved training data\n")

end_loop = time.time()
print("Total time: ", np.round(end_loop - loop_time, 1))
