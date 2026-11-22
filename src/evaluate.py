"""
Evaluate the best saved model on the validation set.
Reports accuracy, per-class metrics, and confusion matrix.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import torch

import config
from dataset import create_data_loaders
from model import load_checkpoint


def plot_confusion_matrix(cm, class_names, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True", xlabel="Predicted",
           title="Confusion Matrix — Validation Set")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontweight="bold")
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


@torch.no_grad()
def main():
    print("=" * 60)
    print("   ISL MODEL EVALUATION")
    print("=" * 60)

    if not config.BEST_MODEL_PATH.exists():
        print(f"[ERROR] No checkpoint at {config.BEST_MODEL_PATH}. Train first.")
        sys.exit(1)

    model, ckpt = load_checkpoint(config.BEST_MODEL_PATH, config.DEVICE)
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch','?')}, device={config.DEVICE})")
    print(f"Classes: {ckpt.get('target_classes', config.TARGET_CLASSES)}\n")

    _, val_loader, _, _ = create_data_loaders(config.BATCH_SIZE)

    all_preds, all_labels = [], []
    use_amp = config.DEVICE.type == "cuda"

    for imgs, labels in val_loader:
        imgs = imgs.to(config.DEVICE)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(imgs)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean() * 100
    print(f"Overall Accuracy: {acc:.2f}%\n")

    names = config.TARGET_CLASSES
    print(classification_report(all_labels, all_preds, target_names=names, digits=4, zero_division=0))

    print("Per-class accuracy:")
    for i, name in enumerate(names):
        mask = all_labels == i
        if mask.sum():
            c_acc = (all_preds[mask] == i).mean() * 100
            print(f"  {name:8s}: {c_acc:6.2f}% ({mask.sum()} samples)")

    cm = confusion_matrix(all_labels, all_preds, labels=range(len(names)))
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = config.RESULTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(cm, names, cm_path)
    print(f"\nConfusion matrix saved: {cm_path}")


if __name__ == "__main__":
    main()
