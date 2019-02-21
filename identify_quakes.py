""" step #1

The initial script reads through all of the labelled data searching for earthquakes (denoted by 'time_to_faulure' reaching zero), 
which are detected by a large positive change in time left.

The purpose here is to create a record of these earthquakes for further data manipulation without having to run this scan again.
Having a record of the earthquakes allows for downstream data manipulation without repeating this work over again.


disecting the CSV training data into numpy arrays for further manipulation
 starting with dividing training data into different earthquake sequences

# cycle through data in chunks of 5million 
# create earthquake dictionary {names = quake # : tuple of which data array and location within that array}
# save quake dict at the end
"""

import numpy as np
import pandas as pd

import os, time, pickle, sys, gc
from tqdm import tqdm

from numba import jit, njit

gc.collect()

np.set_printoptions(precision=15,  linewidth=160, threshold=10_000)
np.random.seed(seed=5)
pd.set_option('display.precision', 28, 'display.max_columns', 6)#, 'display.width', 160)

### small speed boost
@njit
def differentiate(vector):	
	return np.diff(vector)

data_reader = pd.read_csv('/home/brian/ML_Projects/Quake/train/train.csv', dtype={'acoustic_data':np.int64, 'time_to_failure':np.float64}, chunksize=5_000_000)

quakes = {}
num_quake = 0

for count, chunk in tqdm(enumerate(data_reader)):
	# read in values, save as two arrays
	time 		= chunk['time_to_failure'].values

	quake = 0
	quake = np.argwhere(differentiate(time) > 1 )
	if quake != 0:
		num_quake += 1
		quakes[f'{num_quake}'] =(count, quake)


print("Total quakes: ", num_quake)
print(quakes)

pickle_out = open('/home/brian/ML_Projects/Quake/train/quakes.pickle', 'wb')
pickle.dump(quakes, pickle_out)
pickle_out.close()