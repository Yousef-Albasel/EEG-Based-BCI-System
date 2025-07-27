import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt ,iirnotch
from config import *

def load_trial(data_path,row, dir):
    path = os.path.join(data_path, row['task'], dir, row['subject_id'], str(row['trial_session']), 'EEGdata.csv')
    if not os.path.exists(path): return None
    trial = pd.read_csv(path)
    idx = (row['trial'] - 1) * SAMPLES_PER_TRIAL
    start = idx + 500  # Skip first 2s
    return trial.iloc[start:start + SAMPLES_USED]  # Use 4s (1000 samples)


def preprocess_eeg(data, fs=250, band=(5, 40), notch_freqs=[50, 60], harmonics=2):
    nyq = 0.5 * fs
    b, a = butter(4, [band[0]/nyq, band[1]/nyq], btype='band')
    for ch in EEG_CHANNELS:
        data[ch] = filtfilt(b, a, data[ch])
        for f in notch_freqs:
            for h in range(1, harmonics+1):
                nf = f * h
                if nf < nyq:
                    bn, an = iirnotch(nf/nyq, Q=30)
                    data[ch] = filtfilt(bn, an, data[ch])
        data[ch] = (data[ch] - data[ch].mean()) / (data[ch].std() + 1e-8)
    return data
