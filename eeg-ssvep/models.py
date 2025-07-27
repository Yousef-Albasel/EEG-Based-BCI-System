import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from config import *
import torch
import torch.nn as nn

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class MLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes))

    def forward(self, x):
        return self.net(x)

    
def load_model(model_path):
    model = MLP(64, 4)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model