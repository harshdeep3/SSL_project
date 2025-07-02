import torch
import torchvision
from torch.utils.data import DataLoader
from getCifar10Data import CIFAR10Pairs
from simclrModel import SimCLRModel
import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train():
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_path = ".\\SSL_project\\test_data"
    is_training = True
    batch_size = 4
    num_workers = 2
    lr = 1e-4
    epochs = 100

    # get data and convert to dataloader
    dataset = CIFAR10Pairs(root=file_path, is_training=is_training)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # get model
    model = SimCLRModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for image, label in loader:
            image, label = image.to(device), label.to(device)

            plt.imshow(torchvision.utils.make_grid(image).permute(1, 2, 0) / 2 + 0.5)
            plt.show()

            break
        break


if __name__ == "__main__":

    train()
