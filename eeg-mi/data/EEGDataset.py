"""
This class implements the creations of the EEG Dataset, the trials are gathered from the csv files, 
preprocessed and features are extracted using this dataset class
"""

from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from scipy.stats import skew, kurtosis

import numpy as np
import pandas as pd
import pywt

class EEGDataset(Dataset):
    def __init__(self, df, base_path, preprocessor, sfreq=250, nperseg=256, has_labels=True):
        self.df = df
        self.base_path = base_path
        self.preprocessor = preprocessor
        self.sfreq = sfreq
        self.nperseg = nperseg
        self.has_labels = has_labels
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        trial_data = self.load_trial_data(row)
        
        # Preprocess
        eeg = self.preprocessor.preprocess_trial(trial_data, ch_names=['FZ', 'C3', 'CZ', 'C4', 'PZ'])

        # Compute PSD
        psd_features = self.compute_psd_features(eeg)
        if self.has_labels and 'label' in row:
            label = 0 if row['label'].lower() == 'left' else 1
            return psd_features, label
        else:
            return psd_features
            
    def compute_psd_features(self, eeg):
        features = []
        for ch in range(eeg.shape[1]):
            freqs, psd = welch(eeg[:, ch], fs=self.sfreq, nperseg=self.nperseg)
            band_psd = psd[(freqs >= 8) & (freqs <= 30)]
            features.append(band_psd)
        return np.concatenate(features)

    def load_trial_data(self, row):
        id_num = row['id']
        if id_num <= 4800:
            dataset = 'train'
        elif id_num <= 4900:
            dataset = 'validation'
        else:
            dataset = 'test'

        eeg_path = f"{self.base_path}/{row['task']}/{dataset}/{row['subject_id']}/{row['trial_session']}/EEGdata.csv"
        eeg_data = pd.read_csv(eeg_path)
        trial_num = int(row['trial'])
        samples_per_trial = 2250 if row['task'] == 'MI' else 1750

        start_idx = (trial_num - 1) * samples_per_trial
        end_idx = start_idx + samples_per_trial

        eeg_channels = ['FZ', 'C3', 'CZ', 'C4', 'PZ']
        trial_data = eeg_data.iloc[start_idx:end_idx][eeg_channels]
        return trial_data
