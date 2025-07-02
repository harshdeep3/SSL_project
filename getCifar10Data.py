import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset


class SimCLRTransform:
    def __init__(self, image_size=32):
        self.transform = T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class CIFAR10Pairs(Dataset):
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
        x_i, x_j = self.transform(img)
        return x_i, x_j
