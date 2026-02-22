import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def one_hot_encode(sequence: str) -> np.ndarray:
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.zeros((4, len(sequence)), dtype=np.float32)
    for i, nuc in enumerate(sequence.upper()):
        if nuc in mapping:
            encoded[mapping[nuc], i] = 1.0
    return encoded


class GRNAEfficiencyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv1d(4, 64, kernel_size=3, padding=1)
        self.conv2   = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3   = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc1     = nn.Linear(256, 128)
        self.fc2     = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze(-1)


def load_model(model_path: str) -> GRNAEfficiencyPredictor:
    model = GRNAEfficiencyPredictor()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def predict(model: GRNAEfficiencyPredictor, sequence: str) -> float:
    encoded = one_hot_encode(sequence)
    tensor  = torch.tensor(encoded).unsqueeze(0)  # (1, 4, 30)
    with torch.no_grad():
        score = model(tensor).item()
    return round(score, 4)