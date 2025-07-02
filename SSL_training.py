import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from getCifar10Data import CIFAR10Pairs
from simclrModel import SimCLRModel, save_model, load_model
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm


def imshow(img):
    plt.imshow(torchvision.utils.make_grid(img).permute(1, 2, 0) / 2 + 0.5)
    plt.show()


def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    Computes the NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    Pulls positive pairs together and pushes negative pairs apart.

    Args:
        z_i (_type_): embeddings from view 1 (batch_size x dim)
        z_j (_type_): embeddings from view 2 (batch_size x dim)
        temperature (float, optional): scaling factor for logits. Defaults to 0.5.

    Returns:
        Scalar NT-Xent loss value
    """
    # These z_i and z_j are L2-normalized vectors in latent space
    N = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # 2N x D
    # Compute pairwise cosine similarity (2N x 2N)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    sim /= temperature

    # Mask to remove similarity with itself
    labels = torch.arange(N).repeat(2)
    labels = labels.to(z.device)
    mask = torch.eye(2 * N, dtype=torch.bool).to(z.device)

    sim = sim.masked_fill(mask, -9e15)  # large negative to ignore self-similarity
    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)

    loss = F.cross_entropy(sim, targets)
    return loss


def visualize_embeddings(encoder, num_samples=1000):
    """
    Extract and visualize image embeddings using t-SNE.

    Args:
        encoder: trained encoder network (ResNet18)
        num_samples: number of samples to visualize
    """
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder.eval()
    encoder.to(device)

    transform = T.Compose([T.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root="\\SSL_project\\test_data", train=False, download=True, transform=transform
    )
    loader = DataLoader(testset, batch_size=256, shuffle=False)

    all_features = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            features = encoder(x).cpu().numpy()
            all_features.append(features)
            all_labels.append(y.numpy())
            if len(all_features) * 256 >= num_samples:
                break

    features = np.concatenate(all_features, axis=0)[:num_samples]
    labels = np.concatenate(all_labels, axis=0)[:num_samples]

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=300)
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=10)
    plt.legend(
        *scatter.legend_elements(),
        title="Classes",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.title("t-SNE of CIFAR-10 Embeddings (SimCLR Encoder)")
    plt.tight_layout()
    plt.show()


def train(file_path: str):
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_training = True
    if is_training:
        data_file_path = file_path + "train_data"
    else:
        data_file_path = file_path + "test_data"

    batch_size = 4
    num_workers = 2
    lr = 1e-4
    epochs = 10
    temperature = 0.5

    # Load dataset with SSL pairs
    # get data and convert to dataloader
    dataset = CIFAR10Pairs(root=data_file_path, is_training=is_training)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # get model
    model = SimCLRModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0

        for image_i, image_j in loader:
            # two different “views” of the same original image
            image_i, image_j = image_i.to(device), image_j.to(device)
            # image_i and image_j are L2-normalized vectors in latent space
            normalised_image_i, z_j = model(image_i), model(image_j)

            # show images
            # imshow(image)

            # Contrastive loss
            loss = nt_xent_loss(normalised_image_i, z_j, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    save_model(filepath=file_path + "model\\model.pt", model=model)

    return model.encoder  # Return the trained encoder


if __name__ == "__main__":

    folder_path = ".\\SSL_project\\"

    if not os.path.isfile(folder_path + "model\\model.pt"):
        encoder = train(folder_path)
    else:
        encoder = load_model(folder_path + "model\\model.pt").encoder

    visualize_embeddings(encoder, num_samples=1000)
