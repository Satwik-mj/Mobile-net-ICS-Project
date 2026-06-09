import os
import pickle
import numpy as np
from PIL import Image

CIFAR_DIR = r"C:\Users\majum\OneDrive\Desktop\MobileNet_Project\cifar-10-batches-py"
OUT_DIR = "./my_custom_dataset"

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

def unpickle(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    return data

def save_images(data_dict, split, batch_index=0):
    images = data_dict[b"data"]
    labels = data_dict[b"labels"]

    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    for i, (img_array, label) in enumerate(zip(images, labels)):
        class_name = CIFAR10_CLASSES[label]

        dest_dir = os.path.join(OUT_DIR, split, class_name)
        os.makedirs(dest_dir, exist_ok=True)

        filename = f"{split}_b{batch_index}_{i:05d}.png"
        dest_path = os.path.join(dest_dir, filename)

        Image.fromarray(img_array).save(dest_path)

    print(f"Saved {len(images)} images to {split} (batch {batch_index})")

def main():
    print("=" * 60)
    print("CIFAR-10 to ImageFolder Converter")
    print("=" * 60)

    print("\n[1/2] Processing training batches")

    for batch_num in range(1, 6):
        batch_path = os.path.join(CIFAR_DIR, f"data_batch_{batch_num}")
        data = unpickle(batch_path)
        save_images(data, split="train", batch_index=batch_num)

    print("\n[2/2] Processing test batch")

    data = unpickle(os.path.join(CIFAR_DIR, "test_batch"))
    save_images(data, split="test", batch_index=1)

    print("\n" + "=" * 60)
    print("Conversion Complete")
    print(f"Output root: {os.path.abspath(OUT_DIR)}")

    print("\nResulting structure:")

    for split in ["train", "test"]:
        total = 0
        split_dir = os.path.join(OUT_DIR, split)

        for cls in sorted(os.listdir(split_dir)):
            n = len(os.listdir(os.path.join(split_dir, cls)))
            total += n
            print(f"  {split}/{cls:<12} {n:>5} images")

        print(f"  {'-' * 20} {total:>5} total\n")

if __name__ == "__main__":
    main()