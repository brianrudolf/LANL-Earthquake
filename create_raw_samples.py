""" step #2

This sample creation step runs through the time period preceeding each earthquake and creates data samples of the same 
form as the provided test data. The working theory is that further data wrangling will occur on these data samples, 
but that data from each earthquake time period should be isolated from the other periods. 

This allows for different train and CV strategies based on separating the earthquake sequences to be tested downstream
without re-creating these samples.

This script uses a rolling window / aperture to create more data samples than simply separating the available data into
non-overlapping sequences. Due to this, it is imperative that all of the data from each earthquake sequence be used exclusively 
in either training data or cross validation data. If data samples from multiple quake sources are shuffled together and then split, 
it could lead to data existing within the train and validation set due to the overlapping window.

"""

import numpy as np
import pandas as pd

import os, time, pickle
from tqdm import tqdm
from skimage.util import view_as_windows
from numba import jit, njit

start = time. time()

np.set_printoptions(precision=15,  linewidth=160, threshold=10_000)
np.random.seed(seed=5)
pd.set_option('display.precision', 28, 'display.max_columns', 6)#, 'display.width', 160)


@njit
def create_samples(acoustic, data_len):
	tmp_data = np.zeros((data_len, sample_len), dtype=np.int64)
	
	for s in range(data_len):
		tmp_data[s] = acoustic[s*stride : s*stride + sample_len]	
	
	return tmp_data



# number of samples = floor((length - 150_000)/stride) + 1
sample_len = 150_000
stride = 50_000

quakes_dict = pickle.load(open('/home/brian/ML_Projects/Quake/train/quakes.pickle', 'rb'))

### define the time of each earthquake in the input data
quake_time = []
for value in quakes_dict.values():
	# the placement of the earthquake will be within a given division of 5million
	quake_time.append(5_000_000*value[0] + np.asscalar(value[1]))
# the time each earthquake occurs
print(quake_time)


for i in tqdm(range(len(quake_time))):
	if i == 0:
		# add one to account for the header row in the CSV
		to_skip = 1
		# add one to account for size vs index
		size = quake_time[i] + 1
	else:
		# add previous interval size to the amount to skip
		to_skip += size
		# add one to account for size vs index
		size = quake_time[i] - quake_time[i - 1] 

	# verify line numbers are correct
	print("Reading lines ", to_skip, " to ", (to_skip + size))

	df = pd.read_csv('/home/brian/ML_Projects/Quake/train/train.csv', header=None, names=['acoustic_data', 'time_to_failure'], dtype={'acoustic_data':np.int64, 'time_to_failure':np.float64}, skiprows=to_skip, nrows=size)
	#### view data briefly to ensure correct format
	# print(df.head())
	# print(df.tail())

	# cycle through data points to create input samples of 150_000
	samp_num = int(np.floor((size - sample_len)/stride) + 1)


	label_idexes = np.arange(samp_num) * stride + sample_len - 1
	sample_labels = df['time_to_failure'].iloc[label_idexes]
	
	acoustic_array = df['acoustic_data'].values	
	# sample_data = np.zeros((samp_num, sample_len), dtype=np.int64)
	sample_data = create_samples(acoustic_array, samp_num)

	# for s in range(samp_num):
	# 	sample_data[s] = acoustic_array[s*stride : s*stride + sample_len]
	np.save(f'/home/brian/ML_Projects/Quake/train/data_samples/interval_{i}_X.npy', sample_data)
	np.save(f'/home/brian/ML_Projects/Quake/train/data_samples/interval_{i}_y.npy', sample_labels)