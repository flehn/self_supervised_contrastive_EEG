DATA_PATH_TRAIN = 'SEED/SEED_EEG/Preprocessed_EEG'
Output_path = '/SEED/SEED_EEG/BandPassFiltered/'

import numpy as np
from scipy import signal
import mne
import scipy.io as scio
import os
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import torch
from Plotting import PlotEEG

def load_subject(path):
	subject = scio.loadmat(path,verify_compressed_data_integrity=False)
	trials = list(subject)[3:]
	return subject, trials

def plot_frequencies(subject, trial=0):
	trials = list(subject)[3:]
	for i in range(1, 62):
	  yy = subject[trials[trial]][i]

	  timestep = 1e-3
	  xx = np.fft.fftfreq(yy.size, d=timestep)

	  fft_yy = fft(yy)
	  mask = xx > 0
	  plt.plot(xx[mask], np.abs(fft_yy[mask]), label=i)

	#plt.legend(ncol=8)
	plt.xlim(0,50)
	plt.show() 







def bandpass_filter_1(eeg):
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    a,b = signal.butter(5, [4,47], 'bandpass', analog=True)
    return signal.lfilter(b, a, eeg)

def bandpass_filter2(eeg):
  #info = mne.create_info(62, 200, ch_types='misc', verbose=None)
  #x = mne.io.RawArray(eeg, info, first_samp=0, copy='auto', verbose=None)  #for numpy
  #x = mne.io.Raw(eeg) # for mat
  #x = mne.io.read_raw_fieldtrip(fname, info, data_name='data')
  return mne.filter.filter_data(eeg, sfreq=200, l_freq=4, h_freq=47, verbose=False)


def apply_bandpass(input_path='/'):
	# Iterate over all subjects and apply bandpass filter
	for subject in os.listdir(input_path):
	    if subject.endswith(".mat") and not subject.endswith("label.mat"):
	        print(subject)
	        subject_n = os.path.join(input_path, subject)
	        subject_mat = scio.loadmat(subject_n)
	        for matrices in list(subject_mat)[3:]:
	            subject_mat[matrices]  = bandpass_filter2(subject_mat[matrices])

	        filename = str(subject)
	        
	        #print(f'subject_mat: {subject_mat}')
	        scio.savemat(filename, subject_mat, appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')
	        print(f'saved: {filename}')


subject_n = '/Users/fvl/Desktop/Preprocessed_EEG/1_20131027.mat'
subject_mat = scio.loadmat(subject_n)
print(list(subject_mat)[3:])
print(subject_mat['djc_eeg9'].shape)

plot_frequencies(subject_mat, trial=8)

#Plot Channels:

newplot = PlotEEG()
newplot(subject_mat['djc_eeg9'])

for matrices in list(subject_mat)[3:]:
    subject_mat[matrices]  = bandpass_filter2(subject_mat[matrices])

plot_frequencies(subject_mat, trial=8)
newplot2 = PlotEEG()
newplot2(subject_mat['djc_eeg9'])








