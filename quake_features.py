""" 
utility script with functions for generating features
"""

import numpy as np
import os, time, pickle
from tqdm import tqdm

from pyculib.fft import fft
from numba import jit, njit, prange
import numba

eps = 1E-100


def fft_features(data_arr):
	out_data = np.zeros((data_arr.shape[0], 500))

	for i in range(out_data.shape[0]):
		out_data[i] = fft_data(data_arr[i])

	return out_data

def fft_data(data):
	fft_tmp = np.empty((data.shape), dtype=np.complex64)
	yf = numba.cuda.to_device(fft_tmp)
	fft(data.astype(np.complex64), yf)
	tmp = np.copy(np.abs(yf[1:len(yf)//6 + 1]))
	avg_fft = np.mean(tmp.reshape(-1, len(tmp)//500), axis=1)
	return avg_fft


@njit(parallel = True)
def stats(array):
	if len(array) == 0:
		return 0, 0, 0, 0, 0, 0
	else:
		return np.mean(array), np.median(array), np.std(array), np.max(array), np.min(array), np.var(array)	

@njit(parallel = True)
def log_stats(array):
	if len(array) == 0:
		return 0, 0, 0, 0, 0, 0
	else:
		### epsilon values are added to avoid causing negative infinity results from taking log(0)
		return np.log(np.mean(array) + eps), np.log(np.median(array) + eps), np.log(np.std(array) + eps), np.log(np.max(array) + eps), np.log(np.max(array) - np.min(array) + eps), np.log(np.var(array) + eps)	

@njit(parallel = True)
def exp_stats(array):
	if len(array) == 0:
		return 0, 0, 0, 0, 0, 0
	else:
		return np.exp(np.mean(array)), np.exp(np.median(array)), np.exp(np.std(array)), np.exp(np.max(array)), np.exp(np.min(array)), np.exp(np.var(array))	


@njit(parallel = True)
def feature_extraction_172(sample_data):
	data = np.zeros((sample_data.shape[0], 172))

	for i in prange(sample_data.shape[0]):		
		acoustic = sample_data[i]
		
		### create stats for overall data sample		
		sample_mean, 	sample_med, 	sample_std, 	sample_max, 	sample_min, 	sample_var 		= stats(acoustic)
		l_sample_mean, 	l_sample_med, 	l_sample_std, 	l_sample_max, 	l_sample_min, 	l_sample_var 	= log_stats(acoustic)


		### create stats for sections of data sample (divide by percentiles)		
		## first quartile values
		quartile_1 		= np.percentile(acoustic, 25)		
		quar_acoustic_1	= acoustic[acoustic < quartile_1]

		quar_1_mean,  	quar_1_med, 	quar_1_std, 	quar_1_max, 	quar_1_min, 	quar_1_var 		= stats(quar_acoustic_1)
		l_quar_1_mean, 	l_quar_1_med, 	l_quar_1_std, 	l_quar_1_max, 	l_quar_1_min, 	l_quar_1_var 	= log_stats(quar_acoustic_1)


		## second quartile values		
		quar_tmp_2 		= acoustic[acoustic > quartile_1]
		quar_acoustic_2 = quar_tmp_2[quar_tmp_2 < sample_med]

		quar_2_mean, 	quar_2_med, 	quar_2_std, 	quar_2_max, 	quar_2_min, 	quar_2_var 		= stats(quar_acoustic_2)
		l_quar_2_mean, 	l_quar_2_med, 	l_quar_2_std, 	l_quar_2_max, 	l_quar_2_min, 	l_quar_2_var 	= log_stats(quar_acoustic_2)
		
		## third quartile values	
		quartile_3 		= np.percentile(acoustic, 75)		
		quar_tmp_3 		= acoustic[acoustic > sample_med]
		quar_acoustic_3 = quar_tmp_3[quar_tmp_3 < quartile_3]	

		quar_3_mean, 	quar_3_med, 	quar_3_std, 	quar_3_max, 	quar_3_min, 	quar_3_var 		= stats(quar_acoustic_3)
		l_quar_3_mean, 	l_quar_3_med, 	l_quar_3_std, 	l_quar_3_max, 	l_quar_3_min, 	l_quar_3_var 	= log_stats(quar_acoustic_3)
		
		## fourth quartile values		
		quar_acoustic_4 = acoustic[acoustic > quartile_3]
		
		quar_4_mean, 	quar_4_med, 	quar_4_std, 	quar_4_max, 	quar_4_min, 	quar_4_var 		= stats(quar_acoustic_4)
		l_quar_4_mean, 	l_quar_4_med, 	l_quar_4_std, 	l_quar_4_max, 	l_quar_4_min, 	l_quar_4_var 	= log_stats(quar_acoustic_4)
		




		### create stats for sections of data sample (divide by quarters of values min-mean-max)		
		## first quarter values		
		quar_v_1 			= (sample_mean + sample_min)/2
		quar_v_acoustic_1	= acoustic[acoustic < quar_v_1]

		quar_v_1_mean, 		quar_v_1_med, 		quar_v_1_std, 		quar_v_1_max, 		quar_v_1_min, 		quar_v_1_var 	= stats(quar_v_acoustic_1)
		l_quar_v_1_mean,	l_quar_v_1_med, 	l_quar_v_1_std, 	l_quar_v_1_max, 	l_quar_v_1_min, 	l_quar_v_1_var 	= log_stats(quar_v_acoustic_1)

		## second quarter vues
		quar_v_tmp_2 		= acoustic[acoustic > quar_v_1]
		quar_v_acoustic_2	= quar_v_tmp_2[quar_v_tmp_2 < sample_mean]

		quar_v_2_mean, 		quar_v_2_med, 		quar_v_2_std, 		quar_v_2_max, 		quar_v_2_min, 		quar_v_2_var 	= stats(quar_v_acoustic_2)
		l_quar_v_2_mean,	l_quar_v_2_med, 	l_quar_v_2_std, 	l_quar_v_2_max, 	l_quar_v_2_min, 	l_quar_v_2_var 	= log_stats(quar_v_acoustic_2)

		## third quarter vues
		quar_v_3 			= (sample_max + sample_mean)/2	
		quar_v_tmp_3 		= acoustic[acoustic > sample_mean]
		quar_v_acoustic_3	= quar_v_tmp_3[quar_v_tmp_3 < quar_v_3]

		quar_v_3_mean, 		quar_v_3_med, 		quar_v_3_std, 		quar_v_3_max, 		quar_v_3_min, 		quar_v_3_var 	= stats(quar_v_acoustic_3)
		l_quar_v_3_mean,	l_quar_v_3_med, 	l_quar_v_3_std, 	l_quar_v_3_max, 	l_quar_v_3_min, 	l_quar_v_3_var 	= log_stats(quar_v_acoustic_3)
		
		## fourth quarter vues			
		quar_v_acoustic_4 	= acoustic[acoustic > quar_v_3]
		
		quar_v_4_mean, 		quar_v_4_med, 		quar_v_4_std, 		quar_v_4_max, 		quar_v_4_min, 		quar_v_4_var 	= stats(quar_v_acoustic_4)
		l_quar_v_4_mean,	l_quar_v_4_med, 	l_quar_v_4_std, 	l_quar_v_4_max, 	l_quar_v_4_min, 	l_quar_v_4_var 	= log_stats(quar_v_acoustic_4)


		## positive values
		pos_acoustic 	= acoustic[acoustic > 0]		
		
		pos_mean, 	pos_med, 	pos_std, 	pos_max, 	pos_min, 	pos_var 	= stats(pos_acoustic)
		l_pos_mean,	l_pos_med, 	l_pos_std, 	l_pos_max, 	l_pos_min, 	l_pos_var 	= log_stats(pos_acoustic)
		
		## negative values
		neg_acoustic 	= acoustic[acoustic < 0]	
		
		neg_mean, 	neg_med, 	neg_std, 	neg_max, 	neg_min, 	neg_var 	= stats(neg_acoustic)
		l_neg_mean,	l_neg_med, 	l_neg_std, 	l_neg_max, 	l_neg_min, 	l_neg_var 	= log_stats(neg_acoustic)

		## abs values
		abs_acoustic = np.abs(acoustic)		
		
		abs_mean, 	abs_med, 	abs_std, 	abs_max, 	abs_min, 	abs_var 	= stats(abs_acoustic)
		l_abs_mean,	l_abs_med, 	l_abs_std, 	l_abs_max, 	l_abs_min, 	l_abs_var 	= log_stats(abs_acoustic)

		### calculate percentiles
		perc_1	= np.percentile(acoustic, 1)
		perc_2	= np.percentile(acoustic, 2)
		perc_3	= np.percentile(acoustic, 3)
		perc_4	= np.percentile(acoustic, 4)
		perc_5	= np.percentile(acoustic, 5)

		perc_10	= np.percentile(acoustic, 10)
		perc_20	= np.percentile(acoustic, 20)
		perc_30	= np.percentile(acoustic, 30)
		perc_40	= np.percentile(acoustic, 40)

		perc_60	= np.percentile(acoustic, 60)
		perc_70	= np.percentile(acoustic, 70)
		perc_80	= np.percentile(acoustic, 80)
		perc_90	= np.percentile(acoustic, 90)

		

		data[i] = np.array((
														sample_mean,		sample_med, 		sample_std, 		sample_max, 		sample_min, 		sample_var,
														l_sample_mean,		l_sample_med, 		l_sample_std, 		l_sample_max, 		l_sample_min, 		l_sample_var,
			
			quartile_1, 		len(quar_acoustic_1), 	quar_1_mean,		quar_1_med, 		quar_1_std, 		quar_1_max, 		quar_1_min, 		quar_1_var,
														l_quar_1_mean,		l_quar_1_med, 		l_quar_1_std, 		l_quar_1_max, 		l_quar_1_min, 		l_quar_1_var,
			
								len(quar_acoustic_2), 	quar_2_mean,		quar_2_med, 		quar_2_std, 		quar_2_max, 		quar_2_min, 		quar_2_var,
														l_quar_2_mean,		l_quar_2_med, 		l_quar_2_std, 		l_quar_2_max, 		l_quar_2_min, 		l_quar_2_var,
			
			quartile_3, 		len(quar_acoustic_3), 	quar_3_mean,		quar_3_med, 		quar_3_std, 		quar_3_max, 		quar_3_min, 		quar_3_var,
														l_quar_3_mean,		l_quar_3_med, 		l_quar_3_std, 		l_quar_3_max, 		l_quar_3_min, 		l_quar_3_var,
			
						 		len(quar_acoustic_4), 	quar_4_mean,		quar_4_med, 		quar_4_std, 		quar_4_max, 		quar_4_min, 		quar_4_var,
						 								l_quar_4_mean,		l_quar_4_med, 		l_quar_4_std, 		l_quar_4_max, 		l_quar_4_min, 		l_quar_4_var,
			
			quar_v_1,			len(quar_v_acoustic_1),	quar_v_1_mean, 		quar_v_1_med, 		quar_v_1_std, 		quar_v_1_max, 		quar_v_1_min, 		quar_v_1_var,
														l_quar_v_1_mean, 	l_quar_v_1_med, 	l_quar_v_1_std, 	l_quar_v_1_max, 	l_quar_v_1_min, 	l_quar_v_1_var,
			
								len(quar_v_acoustic_2),	quar_v_2_mean, 		quar_v_2_med, 		quar_v_2_std, 		quar_v_2_max, 		quar_v_2_min, 		quar_v_2_var,
														l_quar_v_2_mean, 	l_quar_v_2_med, 	l_quar_v_2_std, 	l_quar_v_2_max, 	l_quar_v_2_min, 	l_quar_v_2_var,
			
			quar_v_3,			len(quar_v_acoustic_3),	quar_v_3_mean, 		quar_v_3_med, 		quar_v_3_std, 		quar_v_3_max, 		quar_v_3_min, 		quar_v_3_var,
														l_quar_v_3_mean, 	l_quar_v_3_med, 	l_quar_v_3_std, 	l_quar_v_3_max, 	l_quar_v_3_min, 	l_quar_v_3_var,
			
								len(quar_v_acoustic_4),	quar_v_4_mean, 		quar_v_4_med, 		quar_v_4_std, 		quar_v_4_max, 		quar_v_4_min, 		quar_v_4_var,
														l_quar_v_4_mean, 	l_quar_v_4_med, 	l_quar_v_4_std, 	l_quar_v_4_max, 	l_quar_v_4_min, 	l_quar_v_4_var,
			
								len(pos_acoustic), 		pos_mean,			pos_med, 			pos_std, 			pos_max, 			pos_min, 			pos_var, 
														l_pos_mean,			l_pos_med, 			l_pos_std, 			l_pos_max, 			l_pos_min, 			l_pos_var, 
			
								len(neg_acoustic), 		neg_mean,			neg_med, 			neg_std, 			neg_max, 			neg_min, 			neg_var, 
														l_neg_mean,			l_neg_med, 			l_neg_std, 			l_neg_max, 			l_neg_min, 			l_neg_var, 
		
								len(abs_acoustic), 		abs_mean,			abs_med, 			abs_std, 			abs_max, 			abs_min, 			abs_var,
														l_abs_mean,			l_abs_med, 			l_abs_std, 			l_abs_max, 			l_abs_min, 			l_abs_var,
			
								perc_1,					perc_2,				perc_3,				perc_4,				perc_5,				perc_10, 			perc_20, 
								perc_30, 				perc_40, 			perc_60, 			perc_70, 			perc_80, 			perc_90
			), dtype=np.float64)
	
	return data



@njit(parallel = True)
def feature_extraction_87(sample_data):
	data = np.zeros((sample_data.shape[0], 87))

	for i in prange(sample_data.shape[0]):		
		acoustic = sample_data[i]
		
		### create stats for overall data sample		
		sample_mean, sample_med, sample_std, sample_max, sample_min, sample_var = stats(acoustic)		


		### create stats for sections of data sample (divide by percentiles)		
		## first quartile values
		quartile_1 		= np.percentile(acoustic, 25)		
		quar_acoustic_1	= acoustic[acoustic < quartile_1]

		quar_1_mean,  quar_1_med, quar_1_std, quar_1_max, quar_1_min, quar_1_var = stats(quar_acoustic_1)


		## second quartile values		
		quar_tmp_2 		= acoustic[acoustic > quartile_1]
		quar_acoustic_2 = quar_tmp_2[quar_tmp_2 < sample_med]

		quar_2_mean, quar_2_med, quar_2_std, quar_2_max, quar_2_min, quar_2_var = stats(quar_acoustic_2)

		
		## third quartile values	
		quartile_3 		= np.percentile(acoustic, 75)		
		quar_tmp_3 		= acoustic[acoustic > sample_med]
		quar_acoustic_3 = quar_tmp_3[quar_tmp_3 < quartile_3]	

		quar_3_mean, quar_3_med, quar_3_std, quar_3_max, quar_3_min, quar_3_var = stats(quar_acoustic_3)

		
		## fourth quartile values		
		quar_acoustic_4 = acoustic[acoustic > quartile_3]
		
		quar_4_mean, quar_4_med, quar_4_std, quar_4_max, quar_4_min, quar_4_var = stats(quar_acoustic_4)
		




		### create stats for sections of data sample (divide by quarters of values min-mean-max)		
		## first quarter values		
		quar_v_1 			= (sample_mean + sample_min)/2
		quar_v_acoustic_1	= acoustic[acoustic < quar_v_1]

		quar_v_1_mean, quar_v_1_med, quar_v_1_std, quar_v_1_max, quar_v_1_min, quar_v_1_var = stats(quar_v_acoustic_1)


		## second quarter vues
		quar_v_tmp_2 		= acoustic[acoustic > quar_v_1]
		quar_v_acoustic_2	= quar_v_tmp_2[quar_v_tmp_2 < sample_mean]

		quar_v_2_mean, quar_v_2_med, quar_v_2_std, quar_v_2_max, quar_v_2_min, quar_v_2_var = stats(quar_v_acoustic_2)


		## third quarter vues
		quar_v_3 			= (sample_max + sample_mean)/2	
		quar_v_tmp_3 		= acoustic[acoustic > sample_mean]
		quar_v_acoustic_3	= quar_v_tmp_3[quar_v_tmp_3 < quar_v_3]

		quar_v_3_mean, quar_v_3_med, quar_v_3_std, quar_v_3_max, quar_v_3_min, quar_v_3_var = stats(quar_v_acoustic_3)

		
		## fourth quarter vues			
		quar_v_acoustic_4 	= acoustic[acoustic > quar_v_3]
		
		quar_v_4_mean, quar_v_4_med, quar_v_4_std, quar_v_4_max, quar_v_4_min, quar_v_4_var = stats(quar_v_acoustic_4)



		## positive values
		pos_acoustic 	= acoustic[acoustic > 0]		
		pos_mean, pos_med, pos_std, pos_max, pos_min, pos_var = stats(pos_acoustic)
		
		## negative values
		neg_acoustic 	= acoustic[acoustic < 0]	
		neg_mean, neg_med, neg_std, neg_max, neg_min, neg_var = stats(neg_acoustic)
		
		## abs values
		abs_acoustic = np.abs(acoustic)		
		abs_mean, abs_med, abs_std, abs_max, abs_min, abs_var = stats(abs_acoustic)


		data[i] = np.array((
														sample_mean,	sample_med, 	sample_std, 	sample_max, 	sample_min, 	sample_var,
			quartile_1, 		len(quar_acoustic_1), 	quar_1_mean,	quar_1_med, 	quar_1_std, 	quar_1_max, 	quar_1_min, 	quar_1_var,
								len(quar_acoustic_2), 	quar_2_mean,	quar_2_med, 	quar_2_std, 	quar_2_max, 	quar_2_min, 	quar_2_var,
			quartile_3, 		len(quar_acoustic_3), 	quar_3_mean,	quar_3_med, 	quar_3_std, 	quar_3_max, 	quar_3_min, 	quar_3_var,
						 		len(quar_acoustic_4), 	quar_4_mean,	quar_4_med, 	quar_4_std, 	quar_4_max, 	quar_4_min, 	quar_4_var,
			quar_v_1,			len(quar_v_acoustic_1),	quar_v_1_mean, 	quar_v_1_med, 	quar_v_1_std, 	quar_v_1_max, 	quar_v_1_min, 	quar_v_1_var,
								len(quar_v_acoustic_2),	quar_v_2_mean, 	quar_v_2_med, 	quar_v_2_std, 	quar_v_2_max, 	quar_v_2_min, 	quar_v_2_var,
			quar_v_3,			len(quar_v_acoustic_3),	quar_v_3_mean, 	quar_v_3_med, 	quar_v_3_std, 	quar_v_3_max, 	quar_v_3_min, 	quar_v_3_var,
								len(quar_v_acoustic_4),	quar_v_4_mean, 	quar_v_4_med, 	quar_v_4_std, 	quar_v_4_max, 	quar_v_4_min, 	quar_v_4_var,
								len(pos_acoustic), 		pos_mean,		pos_med, 		pos_std, 		pos_max, 		pos_min, 		pos_var, 
								len(neg_acoustic), 		neg_mean,		neg_med, 		neg_std, 		neg_max, 		neg_min, 		neg_var, 
								len(abs_acoustic), 		abs_mean,		abs_med, 		abs_std, 		abs_max, 		abs_min, 		abs_var	
			), dtype=np.float64)
	
	return data
