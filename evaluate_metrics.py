import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_fscore_support

from model import build_mobilenet, CustomCrossEntropyLoss
from dataset import get_loaders               

os.makedirs("results", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # <-- ADD THIS LINE FOR SPEED
NUM_CLASSES = 10
EPOCHS_ABLATION = 25
BATCH_SIZE = 256

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def evaluate_per_class(checkpoint_path="mobilenet_alpha_1.0.pth"):
    print("\n" + "="*60)
    print("PART 1 — Per-class metrics (alpha=1.0 trained model)")
    print("="*60)

    _, test_loader, class_names = get_loaders(batch_size=BATCH_SIZE)

    model = build_mobilenet(alpha=1.0, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        digits=4
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(NUM_CLASSES))
    )
    per_class_acc = []
    for cls in range(NUM_CLASSES):
        mask = all_labels == cls
        acc = (all_preds[mask] == all_labels[mask]).mean()
        per_class_acc.append(acc)

    print(report)

    with open("results/per_class_metrics.txt", "w") as f:
        f.write("Per-Class Performance — MobileNetV2 alpha=1.0 on CIFAR-10\n")
        f.write("="*60 + "\n\n")
        f.write(f"{'Class':<14} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}\n")
        f.write("-"*60 + "\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(
                f"{name:<14} {per_class_acc[i]:>10.4f} {precision[i]:>10.4f} "
                f"{recall[i]:>10.4f} {f1[i]:>10.4f} {int(support[i]):>10}\n"
            )
        f.write("\n" + report)

    print("\nSaved → results/per_class_metrics.txt")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(CLASS_NAMES, f1, color="steelblue", edgecolor="black")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score — MobileNetV2 alpha=1.0 on CIFAR-10")
    ax.axhline(y=f1.mean(), color="red", linestyle="--", label=f"Macro avg F1 = {f1.mean():.4f}")
    ax.legend()
    for bar, val in zip(bars, f1):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig("results/per_class_f1.png", dpi=150)
    plt.close()
    print("Saved → results/per_class_f1.png")

    return f1.mean(), precision.mean(), recall.mean()


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def run_ablation():
    print("\n" + "="*60)
    print(f"PART 2 — Ablation study ({EPOCHS_ABLATION} epochs each)")
    print("  Loss A: CustomCrossEntropyLoss (your implementation)")
    print("  Loss B: nn.CrossEntropyLoss   (PyTorch built-in)")
    print("="*60)

    train_loader, test_loader, _ = get_loaders(batch_size=BATCH_SIZE)

    results = {}

    for loss_name, criterion in [
        ("nn.CrossEntropyLoss",          nn.CrossEntropyLoss()),
        ("Custom_CE_no_smoothing",        CustomCrossEntropyLoss(epsilon=0.0, num_classes=10)),
        ("Custom_CE_label_smoothing_0.1", CustomCrossEntropyLoss(epsilon=0.1, num_classes=10)),
    ]:
        print(f"\nTraining with {loss_name} ...")
        torch.manual_seed(42)
        model = build_mobilenet(alpha=1.0, num_classes=NUM_CLASSES).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        epoch_train_acc, epoch_test_acc, epoch_train_loss = [], [], []

        for epoch in range(1, EPOCHS_ABLATION + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            _, te_acc = eval_epoch(model, test_loader, criterion)
            epoch_train_acc.append(tr_acc)
            epoch_test_acc.append(te_acc)
            epoch_train_loss.append(tr_loss)
            print(f"  Epoch {epoch:2d}/{EPOCHS_ABLATION} | "
                  f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.2f}% | "
                  f"Test Acc: {te_acc*100:.2f}%")

        results[loss_name] = {
            "train_acc": epoch_train_acc,
            "test_acc":  epoch_test_acc,
            "train_loss": epoch_train_loss,
            "final_test_acc": epoch_test_acc[-1],
        }

    with open("results/ablation_results.txt", "w") as f:
        f.write("Ablation Study — CustomCrossEntropyLoss vs nn.CrossEntropyLoss\n")
        f.write(f"Architecture: MobileNetV2 alpha=1.0 | Dataset: CIFAR-10 | Epochs: {EPOCHS_ABLATION}\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Epoch':<8}")
        for name in results:
            short = "Custom" if "Custom" in name else "nn.CE"
            f.write(f"{short+' Train':>14} {short+' Test':>13}")
        f.write("\n" + "-"*70 + "\n")
        for ep in range(EPOCHS_ABLATION):
            f.write(f"{ep+1:<8}")
            for name in results:
                tr = results[name]["train_acc"][ep] * 100
                te = results[name]["test_acc"][ep] * 100
                f.write(f"{tr:>13.2f}% {te:>12.2f}%")
            f.write("\n")
        f.write("\n" + "="*70 + "\n")
        for name, res in results.items():
            f.write(f"Final Test Accuracy — {name}: {res['final_test_acc']*100:.2f}%\n")

    print("\nSaved → results/ablation_results.txt")

    epochs = range(1, EPOCHS_ABLATION + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {
        "nn.CrossEntropyLoss":           "darkorange",
        "Custom_CE_no_smoothing":         "steelblue",
        "Custom_CE_label_smoothing_0.1":  "green",
    }
    labels_map = {
        "nn.CrossEntropyLoss":           "nn.CrossEntropyLoss (baseline)",
        "Custom_CE_no_smoothing":         "Custom CE (no smoothing, epsilon=0)",
        "Custom_CE_label_smoothing_0.1":  "Custom CE + Label Smoothing (epsilon=0.1)",
    }

    for name, res in results.items():
        axes[0].plot(epochs, [a*100 for a in res["train_acc"]],
                     label=labels_map[name], color=colors[name],
                     marker="o", markersize=4)
        axes[1].plot(epochs, [a*100 for a in res["test_acc"]],
                     label=labels_map[name], color=colors[name],
                     marker="o", markersize=4)

    axes[0].set_title("Train Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Ablation: nn.CrossEntropyLoss vs Custom CE vs Custom CE + Label Smoothing\nMobileNetV2 alpha=1.0 on CIFAR-10",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/ablation_summary.png", dpi=150)
    plt.close()
    print("Saved → results/ablation_summary.png")

    print("\n" + "="*60)
    print("FINAL COMPARISON:")
    for name, res in results.items():
        print(f"  {name}: {res['final_test_acc']*100:.2f}% test accuracy")
    print("="*60)


if __name__ == "__main__":
    print("Running evaluation on trained model...")
    evaluate_per_class(checkpoint_path="mobilenet_alpha_1.0.pth")

    
    run_ablation()

    print("\n✓ All results saved to results/ folder")
    print("  results/per_class_metrics.txt")
    print("  results/per_class_f1.png")
    print("  results/ablation_results.txt")
    print("  results/ablation_summary.png")