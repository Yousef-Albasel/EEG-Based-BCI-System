BASE_PATH = ''
SAMPLING_RATE = 250
SAMPLES_PER_TRIAL = 1750  # Full trial length
SAMPLES_USED = 1000       # Only use 4 seconds (250 Hz * 4)
EEG_CHANNELS = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
SSVEP_FREQS = {'Forward': 7, 'Backward': 8, 'Left': 10, 'Right': 13}
EXTENDED_FILTER_BANK = [(5, 15), (6, 14), (10, 20), (14, 22), (18, 28), (22, 30), (25, 35), (30, 38)]

FEATURES_PATH = 'data/features/cached_features.npz'