"""
Training pipeline for ISL word recognition.

Two-stage transfer learning on EfficientNet-B0:
  Stage 1 — Frozen backbone, train classifier head only.
  Stage 2 — Unfreeze all layers, fine-tune with CosineAnnealingLR.

Key design choices:
  • Training criterion uses mild label smoothing (regularisation).
  • Validation criterion uses NO smoothing (true cross-entropy for honest measurement).
  • AMP (float16) for GPU speed and memory savings.
  • Early stopping on val loss.
  • Best-val-loss checkpoint saving (never overwrites with worse).
"""

import sys
import gc
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from dataset import create_data_loaders
from model import build_model, freeze_backbone, unfreeze_all, count_params, save_checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# Training & evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = device.type == "cuda"

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total * 100


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = device.type == "cuda"

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total * 100


def plot_curves(history: Dict[str, List[float]], path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], "o-", label="Train")
    ax1.plot(epochs, history["val_loss"],   "s-", label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "o-", label="Train")
    ax2.plot(epochs, history["val_acc"],   "s-", label="Val")
    ax2.set_title("Accuracy (%)"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("%")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("     ISL WORD RECOGNITION — TRAINING PIPELINE")
    print("=" * 65)
    device = config.DEVICE
    print(f"Device       : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM         : {vram:.1f} GB")
    print(f"Architecture : {config.MODEL_NAME}")
    print(f"Classes ({config.NUM_CLASSES}) : {config.TARGET_CLASSES}")
    print(f"Batch size   : {config.BATCH_SIZE}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    try:
        train_loader, val_loader, c2i, i2c = create_data_loaders(config.BATCH_SIZE)
    except Exception as e:
        print(f"[FATAL] Dataset error: {e}"); sys.exit(1)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(pretrained=True).to(device)

    # Criteria: training has smoothing, validation does NOT
    train_criterion = nn.CrossEntropyLoss(label_smoothing=config.TRAIN_LABEL_SMOOTHING)
    val_criterion   = nn.CrossEntropyLoss(label_smoothing=config.VAL_LABEL_SMOOTHING)

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch = 0
    global_epoch = 0

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1 — Frozen backbone, train classifier head
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 65)
    print(f"STAGE 1: Classifier head only ({config.STAGE1_EPOCHS} epochs, LR={config.STAGE1_LR})")
    print("=" * 65)
    freeze_backbone(model)
    tot, trn = count_params(model)
    print(f"Params: {tot:,} total | {trn:,} trainable ({trn/tot*100:.1f}%)\n")

    opt1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=config.STAGE1_LR, weight_decay=config.WEIGHT_DECAY)

    try:
        for ep in range(1, config.STAGE1_EPOCHS + 1):
            global_epoch += 1
            tl, ta = train_one_epoch(model, train_loader, train_criterion, opt1, scaler, device)
            vl, va = eval_epoch(model, val_loader, val_criterion, device)

            history["train_loss"].append(tl); history["val_loss"].append(vl)
            history["train_acc"].append(ta);  history["val_acc"].append(va)

            tag = ""
            if vl < best_val_loss:
                best_val_loss, best_epoch = vl, global_epoch
                save_checkpoint(model, config.BEST_MODEL_PATH, c2i, i2c,
                                {"val_loss": vl, "val_acc": va, "epoch": global_epoch})
                tag = " ✓ saved"

            print(f"  Epoch {global_epoch:02d} | "
                  f"TL={tl:.4f}  TA={ta:.1f}% | "
                  f"VL={vl:.4f}  VA={va:.1f}%{tag}")

    except torch.cuda.OutOfMemoryError:
        print("[OOM] Reduce BATCH_SIZE in config.py and rerun.")
        gc.collect(); torch.cuda.empty_cache(); sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2 — Full fine-tuning with cosine annealing
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print(f"STAGE 2: Full fine-tune ({config.STAGE2_EPOCHS} epochs, LR={config.STAGE2_LR})")
    print("=" * 65)
    unfreeze_all(model)
    tot, trn = count_params(model)
    print(f"Params: {tot:,} total | {trn:,} trainable ({trn/tot*100:.1f}%)\n")

    opt2 = AdamW(model.parameters(), lr=config.STAGE2_LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(opt2, T_max=config.STAGE2_EPOCHS, eta_min=1e-6)

    patience_counter = 0

    try:
        for ep in range(1, config.STAGE2_EPOCHS + 1):
            global_epoch += 1
            tl, ta = train_one_epoch(model, train_loader, train_criterion, opt2, scaler, device)
            vl, va = eval_epoch(model, val_loader, val_criterion, device)
            scheduler.step()

            history["train_loss"].append(tl); history["val_loss"].append(vl)
            history["train_acc"].append(ta);  history["val_acc"].append(va)

            tag = ""
            if vl < best_val_loss:
                best_val_loss, best_epoch = vl, global_epoch
                patience_counter = 0
                save_checkpoint(model, config.BEST_MODEL_PATH, c2i, i2c,
                                {"val_loss": vl, "val_acc": va, "epoch": global_epoch})
                tag = " ✓ saved"
            else:
                patience_counter += 1

            print(f"  Epoch {global_epoch:02d} | "
                  f"TL={tl:.4f}  TA={ta:.1f}% | "
                  f"VL={vl:.4f}  VA={va:.1f}%{tag}"
                  f"  (LR={scheduler.get_last_lr()[0]:.2e})")

            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n  [EARLY STOP] No improvement for {patience_counter} epochs.")
                break

    except torch.cuda.OutOfMemoryError:
        print("[OOM] Reduce BATCH_SIZE in config.py and rerun.")
        gc.collect(); torch.cuda.empty_cache(); sys.exit(1)

    # ── Save curves ───────────────────────────────────────────────────────
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    curve_path = config.RESULTS_DIR / "training_curves.png"
    plot_curves(history, curve_path)

    print(f"\n{'=' * 65}")
    print(f"TRAINING COMPLETE")
    print(f"  Best val loss : {best_val_loss:.4f}  (epoch {best_epoch})")
    print(f"  Curves saved  : {curve_path}")
    print(f"  Checkpoint    : {config.BEST_MODEL_PATH}")
    print(f"{'=' * 65}")
    print("Next: python evaluate.py")


if __name__ == "__main__":
    main()
