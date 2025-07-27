import torch

BASE_PATH = r'mtcaic3' # Path to the base directory containing the dataset
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CH_NAMES = ['C3', 'CZ', 'C4']
SEED = 42
WINDOW_SIZE = 750
STRIDE = 250
