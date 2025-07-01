"""
This Class implements the preprocessing pipelines for the EEG Data
"""
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
import torch
from sklearn.preprocessing import StandardScaler
from mne.preprocessing import ICA
from collections import defaultdict
import mne
from mne.preprocessing import ICA
import contextlib
import os

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            yield
            
class EEGPreprocessor:
    def __init__(self, sample_rate=250, lowcut=8.0, highcut=30.0, notch_freq=50.0):
        self.sample_rate = sample_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq

    def bandpass_filter(self, data, order=4):
        nyquist = 0.5 * self.sample_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=0)

    def notch_filter(self, data, quality_factor=30):
        b, a = signal.iirnotch(self.notch_freq, quality_factor, self.sample_rate)
        return filtfilt(b, a, data, axis=0)

    def standardize(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1e-6
        return (data - mean) / std


    def crop_epoch(self, data, start_sec=2, end_sec=6):
        start_sample = int(start_sec * self.sample_rate)
        end_sample = int(end_sec * self.sample_rate)
        return data[start_sample:end_sample]
    
    def apply_ica(self, raw_eeg_data, ch_names):
        if ch_names is None:
            raise ValueError("ch_names must be provided for ICA")
        with suppress_stdout():
            # Create MNE Info object
            info = mne.create_info(ch_names=ch_names, sfreq=self.sample_rate, ch_types='eeg')
    
            # Convert numpy to RawArray
            raw = mne.io.RawArray(raw_eeg_data.T, info)
            raw.rename_channels({
                'FZ': 'Fz',
                'CZ': 'Cz',
                'PZ': 'Pz',
                # 'OZ': 'Oz'
            })
    
            raw.set_montage('standard_1020')
    
            # Run ICA
            ica = mne.preprocessing.ICA(n_components=len(ch_names), random_state=97, max_iter="auto")
            ica.fit(raw)
    
            ica.exclude = [0, 1, 2]
            raw_clean = ica.apply(raw.copy())

        return raw_clean.get_data().T  # shape back to (samples, channels)
    
    def baseline_correct(self, data, baseline_start_sec=0.0, baseline_end_sec=2.0):
        start_sample = int(baseline_start_sec * self.sample_rate)
        end_sample = int(baseline_end_sec * self.sample_rate)
        baseline = np.mean(data[start_sample:end_sample], axis=0)
        return data - baseline

    def preprocess_trial(self, trial_data, ch_names=None):
        data = trial_data.values

        data = self.bandpass_filter(data)
        data = self.notch_filter(data)
    
        data = self.apply_ica(data, ch_names) 
        data = self.baseline_correct(data, baseline_start_sec=0.0, baseline_end_sec=2.0)

        data = self.crop_epoch(data, 2.5, 6)
        data = self.standardize(data)
        return data
