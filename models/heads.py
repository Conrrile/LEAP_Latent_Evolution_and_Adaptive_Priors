import torch
import torch.nn as nn
import numpy as np

class Linear_Classifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.fc = nn.Linear(n_features, n_classes)
    def forward(self, x):
        return self.fc(x)

class MLP_Classifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        return self.net(x)

