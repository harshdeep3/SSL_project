import torchvision
import torch.nn as nn


class SimCLRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torchvision.models.resnet18(pretrained=False)
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 64)
        )

    def forward(self, x):
        features = self.encoder(x)
        projection = self.projector(features)
        return projection
