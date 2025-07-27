import numpy as np
from sklearn.cross_decomposition import CCA
from preprocessing.EEGPreprocessor import preprocess_eeg
import mne
from config import *


def generate_ref_signals(freq, n, fs=250, harmonics=4):
    t = np.arange(n) / fs
    ref = [np.sin(2*np.pi*freq*h*t) for h in range(1,harmonics+1)] + [np.cos(2*np.pi*freq*h*t) for h in range(1,harmonics+1)]
    return np.array(ref).T

# --- Extract Features ---
def extract_features(trial):
    data = preprocess_eeg(trial.copy())
    info = mne.create_info(ch_names=EEG_CHANNELS, sfreq=SAMPLING_RATE, ch_types=['eeg'] * len(EEG_CHANNELS))
    raw = mne.io.RawArray(data[EEG_CHANNELS].values.T * 1e-6, info, verbose=False)
    n = trial.shape[0]
    features = []
    for l, h in EXTENDED_FILTER_BANK:
        try:
            band = raw.copy().filter(l_freq=l, h_freq=h, verbose=False)
            sig = band.get_data().T
            corrs = []
            for f in SSVEP_FREQS.values():
                ref = generate_ref_signals(f, n)
                cca = CCA(n_components=1)
                cca.fit(sig, ref)
                x_c, y_c = cca.transform(sig, ref)
                r = np.corrcoef(x_c.T, y_c.T)[0, 1]
                corrs.append(r if not np.isnan(r) else 0)
            features.extend(corrs + [r**2 for r in corrs])
        except:
            features.extend([0] * (len(SSVEP_FREQS) * 2))
    return np.array(features)