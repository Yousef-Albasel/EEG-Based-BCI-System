import torch.nn as nn
import torch.nn.functional as F

class CrossSubjectClassifier(nn.Module):
    def __init__(self, m, num_classes=2, hidden1=128, hidden2=64, dropout_p=0.3):
        super().__init__()
        self.fc1 = nn.Linear(m, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, z):
        x = F.relu(self.bn1(self.fc1(z)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)  # logits
        return x