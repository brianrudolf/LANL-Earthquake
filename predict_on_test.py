""" step #5

This script is configured to read in the test data in the same manner that training data was collected.
A chosen model and scaler are loaded in, and once features are created they are scaled by and fed into 
the model for prediction. These predictions are then stored into a CSV file for submission to Kaggle. 

"""

import numpy as np
import pandas as pd
import os, time, pickle

from tqdm import tqdm
from joblib import dump, load, Parallel, delayed
import multiprocessing

from quake_features import feature_extraction_172, fft_features

np.set_printoptions(threshold=50)

def process_Test(file):
	df = pd.read_csv('test/{}'.format(test_seqs[file]))
	return df['acoustic_data'].values


### Load model
model_name = 'model_SVR_172_stats_folded_rbf_0.002_50_2.259.joblib'
model = load(f'train/numpy_data/5mil/models/{model_name}')

### Load scaler
scaler_name = 'scaler_SVR_172_stats_folded_rbf_0.002_50_2.259.joblib'
scaler = load(f'train/numpy_data/5mil/models/{scaler_name}')

### Create features for test data
test_seqs 	= os.listdir('test')
num_test 	= len(test_seqs)
in_rows  	= 150_000
test_data 	= np.zeros((num_test,	in_rows), dtype=np.float64)
	
start = time.time()

test_data = Parallel(n_jobs=4)(delayed(process_Test)(i) for i in tqdm(range(len(test_seqs))))
test_data = np.vstack(test_data)
print("Test data shape: ", test_data.shape)

stats  = feature_extraction_172(test_data)
ffts   = fft_features(test_data)
X_test = np.concatenate((stats, ffts), axis=1)
# X_test = feature_extraction_172(test_data)

X_test = np.nan_to_num(X_test)
X_test = scaler.transform(X_test)
print("Test features shape: ", X_test.shape)
del test_data

### Make predictions on test data
test_predictions = model.predict(X_test)

### create submission csv
predict_df = pd.DataFrame(data={'seg_id' : list(map(lambda x: x.split('.')[0], test_seqs)), 'time_to_failure' : test_predictions})
print(predict_df.head())

csv_name = model_name.split('.j')[0]
predict_df.to_csv('predictions/{}_{}.csv'.format(csv_name, time.time()), index=False)
