from scipy.signal import butter, filtfilt
import mne
from contextlib import contextmanager
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
import os
from utils import filter_defective_trials
import pandas as pd
from tqdm import tqdm
@contextmanager
def suppress_stdout():
    import sys, os
    with open(os.devnull, 'w') as f:
        old_stdout = sys.stdout
        sys.stdout = f
        try:
            yield
        finally:
            sys.stdout = old_stdout

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
            info = mne.create_info(ch_names=ch_names, sfreq=self.sample_rate, ch_types='eeg')
    
            # Convert numpy to RawArray
            raw = mne.io.RawArray(raw_eeg_data.T, info)
            raw.rename_channels({
                # 'FZ': 'Fz',
                'CZ': 'Cz',
                # 'PZ': 'Pz',
                # 'OZ': 'Oz'
            })
    
            raw.set_montage('standard_1020')
    
            ica = mne.preprocessing.ICA(n_components=len(ch_names),method='fastica', random_state=42, max_iter="auto")
            ica.fit(raw)
    
            ica.exclude = [0, 1]
            raw_clean = ica.apply(raw.copy())

        return raw_clean.get_data().T
    
    def baseline_correct(self, data, baseline_start_sec=0.0, baseline_end_sec=2.0):
        start_sample = int(baseline_start_sec * self.sample_rate)
        end_sample = int(baseline_end_sec * self.sample_rate)
        baseline = np.mean(data[start_sample:end_sample], axis=0)
        return data - baseline

    def preprocess_trial(self, trial_data, ch_names=None):
        if ch_names is not None:
            trial_data = trial_data[ch_names]
        data = trial_data.values

        data = self.bandpass_filter(data)
        data = self.notch_filter(data)
    
        data = self.apply_ica(data, ch_names) 
        data = self.baseline_correct(data, baseline_start_sec=0.0, baseline_end_sec=2.0)
        data = self.crop_epoch(data, 3.5, 7.5)
        data = self.standardize(data)
        
        return data
    
def preprocess_and_save_all(df, base_path, preprocessor, cache_dir='preprocessed_trials', fs=250, ch_names=None, is_test = False):
    ch_names = ['C3', 'CZ', 'C4']
    # ch_names = ['FZ', 'C3', 'CZ', 'C4']
    os.makedirs(cache_dir, exist_ok=True)

    for i in tqdm(range(len(df)), desc="Preprocessing and caching EEG trials"):
        row = df.iloc[i]
        trial_id = row['id']
        if not is_test:
            label = 0 if row['label'].lower() == 'left' else 1
        cache_path = os.path.join(cache_dir, f"trial_{trial_id}.npy")

        if os.path.exists(cache_path):
            continue

        # Determine dataset folder
        id_num = row['id']
        if id_num <= 4800:
            dataset = 'train'
        elif id_num <= 4900:
            dataset = 'validation'
        else:
            dataset = 'test'

        eeg_path = f"{base_path}/{row['task']}/{dataset}/{row['subject_id']}/{row['trial_session']}/EEGdata.csv"
        eeg_data = pd.read_csv(eeg_path)

        samples_per_trial = 2250 if row['task'] == 'MI' else 1750
        trial_num = int(row['trial'])
        start_idx = (trial_num - 1) * samples_per_trial
        end_idx = start_idx + samples_per_trial
        trial_df = eeg_data.iloc[start_idx:end_idx]

        try:
            processed = preprocessor.preprocess_trial(trial_df, ch_names)
            if not is_test:
                np.save(cache_path, {'data': processed, 'label': label})
            else:
                np.save(cache_path, {'data': processed})

        except Exception as e:
            print(f"Failed trial {trial_id}: {e}")

