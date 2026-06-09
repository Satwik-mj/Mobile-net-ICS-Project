import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

TRAIN_DIR = "./my_custom_dataset/train"
TEST_DIR = "./my_custom_dataset/test"

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

def get_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=transform
    )

    class_names = train_dataset.classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader, class_names

if __name__ == "__main__":
    print("=" * 60)
    print("Dataset Pipeline Health Check")
    print("=" * 60)

    train_loader, test_loader, class_names = get_loaders(batch_size=32)

    print(f"\nDetected classes ({len(class_names)} total):")

    for idx, name in enumerate(class_names):
        print(f"  [{idx:>3}] {name}")

    images, labels = next(iter(train_loader))

    print("\nSample training batch:")
    print(f"  images tensor shape : {list(images.shape)}")
    print(f"  labels tensor shape : {list(labels.shape)}")
    print(f"  pixel value range   : [{images.min():.3f}, {images.max():.3f}]")

    print(f"\nTotal training batches : {len(train_loader)}")
    print(f"Total test batches     : {len(test_loader)}")

    print("\nHealth check passed")