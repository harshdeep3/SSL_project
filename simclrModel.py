import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SimCLRModel(nn.Module):
    """
    SimCLR model with an encoder (ResNet18) and a projection head.
    The encoder extracts features, and the projection head maps
    them to a latent space for contrastive learning.
    """

    def __init__(self):
        super().__init__()
        # Base encoder
        self.encoder = torchvision.models.resnet18(pretrained=False)
        # Remove final classification layer
        self.encoder.fc = nn.Identity()
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 64)
        )

    def forward(self, x):
        # Extract features
        h = self.encoder(x)
        # Project to latent space
        z = self.projector(h)
        # Normalize embeddings
        return F.normalize(z, dim=1)


def load_model(filepath: str):
    """
    Load the model

    Args:
        filepath (str): location to saved the model too
    """
    model = SimCLRModel()
    model.load_state_dict(torch.load(filepath, weights_only=True))

    return model


def save_model(filepath: str, model: SimCLRModel):
    """
    Save the model

    Args:
        filepath (str): location to save the model too
        model (SimCLRModel): SimCLR model
    """
    torch.save(model.state_dict(), filepath)
