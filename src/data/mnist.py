from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_mnist_datasets(data_dir: str = "data/raw"):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    full_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset


def get_mnist_dataloaders(
    data_dir: str = "data/raw",
    batch_size: int = 128,
    num_workers: int = 2,
):
    train_dataset, val_dataset, test_dataset = get_mnist_datasets(data_dir=data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader