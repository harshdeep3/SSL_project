import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset


class SimCLRTransform:
    """
    Apply SimCLR-style data augmentations to an input image.
    Returns two differently augmented views of the same image.
    """

    def __init__(self, image_size=32):
        self.transform = T.Compose(
            [
                T.RandomResizedCrop(
                    image_size, scale=(0.2, 1.0)
                ),  # Random crop and resize
                T.RandomHorizontalFlip(),  # Random horizontal flip
                T.RandomApply(
                    [T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),  # Random color jitter
                T.RandomGrayscale(p=0.2),  # Random grayscale
                T.ToTensor(),  # Convert to tensor
            ]
        )

    def __call__(self, x):
        # Return two augmented versions
        return self.transform(x), self.transform(x)


class CIFAR10Pairs(Dataset):
    """
    Custom CIFAR-10 dataset that returns a pair of augmented images
    generated from the same original image for contrastive learning.

    Args:
        root (str): where to store the data
        is_training (bool): true if training, false otherwise
    """

    def __init__(
        self, root: str = ".\\SSL_project\\test_data", is_training: bool = True
    ):
        self.base_dataset = torchvision.datasets.CIFAR10(
            root=root, train=is_training, download=True
        )
        self.transform = SimCLRTransform()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        # two different “views” of the same original image
        x_i, x_j = self.transform(img)
        return x_i, x_j
