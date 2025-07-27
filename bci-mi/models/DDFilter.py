import torch
import torch.nn as nn
import torch.nn.functional as F

class DDFilter(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=16, num_classes=2):
        super().__init__()
        self.reduce = nn.Linear(input_dim, hidden_dim, bias=False)
        self.dd = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=True)

    def forward(self, x):
        g1 = self.reduce(x)              
        g1_sq = g1 * g1                  
        g2 = self.dd(g1_sq)              
        out = self.classifier(g2)        
        return out
