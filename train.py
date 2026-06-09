
import torch
import torch.optim as optim

from dataset import get_loaders
from model import build_mobilenet, CustomCrossEntropyLoss
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def train_model(alpha=1.0, epochs=5, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    train_loader, test_loader, class_names = get_loaders(batch_size=batch_size)

    num_classes = len(class_names)
    print(f"[train] Classes ({num_classes}): {class_names}")

    model = build_mobilenet(
        alpha=alpha,
        num_classes=num_classes
    ).to(device)

    criterion = CustomCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(images)

            loss = criterion(logits, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

            preds = logits.argmax(dim=1)

            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        epoch_loss = running_loss / total_train
        epoch_acc = correct_train / total_train * 100

        print(
            f"\n[Epoch {epoch}] Train Loss: "
            f"{epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%"
        )

        model.eval()

        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(images)
                preds = logits.argmax(dim=1)

                correct_test += (preds == labels).sum().item()
                total_test += labels.size(0)

        test_acc = correct_test / total_test * 100

        print(f"[Epoch {epoch}] Test Acc: {test_acc:.2f}%\n")

    save_path = f"mobilenet_alpha_{alpha}.pth"

    torch.save(model.state_dict(), save_path)

    size_mb = os.path.getsize(save_path) / (1024 ** 2)

    print(
        f"[train] Weights saved to {save_path} "
        f"({size_mb:.2f} MB)"
    )

    return save_path

if __name__ == "__main__":
    print("=" * 60)
    print("Training Standard Model (alpha = 1.00)")
    print("=" * 60)

    train_model(alpha=1.0, epochs=25)

    print("\n" + "=" * 60)
    print("Training Lightweight Model (alpha = 0.35)")
    print("=" * 60)

    train_model(alpha=0.35, epochs=25)