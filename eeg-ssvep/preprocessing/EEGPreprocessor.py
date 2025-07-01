"""
Advanced EEG preprocessing for SSVEP classification
"""

import numpy as np
import pandas as pd
import scipy.signal
from scipy.signal import butter, filtfilt
import mne
from config import EEG_CHANNELS

mne.set_log_level('WARNING')


class SSVEPPreprocessor:
    """
    Advanced preprocessing for SSVEP EEG data
    """
    
    def __init__(self, fs=250, band=(5, 40), notch_freqs=[50, 60], 
                 harmonics=2, car=True, zscore_outlier=True):
        self.fs = fs
        self.band = band
        self.notch_freqs = notch_freqs
        self.harmonics = harmonics
        self.car = car
        self.zscore_outlier = zscore_outlier
        
    def advanced_preprocess_eeg(self, data):
        """
        Apply advanced preprocessing to EEG data
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Raw EEG data
            
        Returns:
        --------
        pandas.DataFrame
            Preprocessed EEG data
        """
        filtered = data.copy()

        # 1. Bandpass filter
        nyq = 0.5 * self.fs
        b, a = butter(4, [self.band[0]/nyq, self.band[1]/nyq], btype='band')

        for ch in EEG_CHANNELS:
            if ch in filtered.columns and filtered[ch].notna().all() and len(filtered[ch].unique()) > 1:
                filtered[ch] = filtfilt(b, a, filtered[ch])

        # 2. Notch filters for power line interference
        for base_freq in self.notch_freqs:
            for h in range(1, self.harmonics + 1):
                notch = base_freq * h
                if notch < nyq:  # Ensure frequency is below Nyquist
                    b_notch, a_notch = scipy.signal.iirnotch(notch / nyq, Q=30)
                    for ch in EEG_CHANNELS:
                        if ch in filtered.columns:
                            filtered[ch] = filtfilt(b_notch, a_notch, filtered[ch])

        # 3. Z-score outlier removal (per channel)
        if self.zscore_outlier:
            for ch in EEG_CHANNELS:
                if ch in filtered.columns:
                    vals = filtered[ch].values
                    z = (vals - np.nanmean(vals)) / (np.nanstd(vals) + 1e-8)
                    vals[np.abs(z) > 5] = np.nanmedian(vals)  # Replace extreme outliers
                    filtered[ch] = vals

        # 4. Channel normalization (zero mean, unit variance)
        for ch in EEG_CHANNELS:
            if ch in filtered.columns:
                vals = filtered[ch].values
                filtered[ch] = (vals - np.nanmean(vals)) / (np.nanstd(vals) + 1e-8)

        # 5. Common Average Reference (CAR)
        if self.car:
            available_channels = [ch for ch in EEG_CHANNELS if ch in filtered.columns]
            if len(available_channels) > 1:
                car_signal = filtered[available_channels].mean(axis=1)
                for ch in available_channels:
                    filtered[ch] = filtered[ch] - car_signal

        return filtered

    def create_mne_raw_object(self, trial_data, srate=250):
        """
        Create MNE Raw object from EEG trial data.
        
        Parameters:
        -----------
        trial_data : pandas.DataFrame
            EEG trial data
        srate : int
            Sampling rate
            
        Returns:
        --------
        mne.io.Raw
            MNE Raw object
        """
        available_channels = [ch for ch in EEG_CHANNELS if ch in trial_data.columns]
        ch_types = ['eeg'] * len(available_channels)

        info = mne.create_info(ch_names=available_channels, sfreq=srate, ch_types=ch_types)
        raw_data = trial_data[available_channels].values.T * 1e-6  # Convert to Volts
        raw = mne.io.RawArray(raw_data, info, verbose=False)

        return raw
