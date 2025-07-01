# Configuration file for SSVEP EEG Classification

# 📁 Dataset configuration
BASE_PATH = ''

# 🎵 SSVEP stimulus frequencies
SSVEP_FREQS = {
    'Forward': 7,   # 7 Hz
    'Backward': 8,  # 8 Hz
    'Left': 10,     # 10 Hz
    'Right': 13     # 13 Hz
}

# 🧠 EEG configuration
SAMPLING_RATE = 250  # Hz
EEG_CHANNELS = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
SAMPLES_PER_TRIAL = 1750  # 7 seconds * 250 Hz

# 🔧 Filter bank configuration for FBCCA
FILTER_BANK = [
    (6, 14),    # Fundamental frequencies
    (14, 22),   # 1st harmonic
    (22, 30),   # 2nd harmonic
    (30, 38)    # 3rd harmonic
]

# Extended filter bank for enhanced features
EXTENDED_FILTER_BANK = [
    (5, 15),    # Low frequency band
    (6, 14),    # Fundamental band
    (10, 20),   # Mid-low band
    (14, 22),   # 1st harmonic
    (18, 28),   # Mid-high band
    (22, 30),   # 2nd harmonic
    (25, 35),   # High band
    (30, 38)    # 3rd harmonic
]

# Model configuration
MODEL_CONFIG = {
    'kernel': 'linear',
    'random_state': 42,
    'probability': True,
    'class_weight': 'balanced'
}

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    'sampling_rate': SAMPLING_RATE,
    'band': (5, 40),
    'notch_freqs': [50, 60],
    'harmonics': 2,
    'car': True,
    'zscore_outlier': True
}
