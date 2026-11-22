"""
EfficientNet-B0 model for ISL word recognition.
Transfer learning with ImageNet pretrained weights.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import config


def build_model(num_classes: int = config.NUM_CLASSES,
                dropout: float = config.DROPOUT_RATE,
                pretrained: bool = True) -> nn.Module:
    """
    EfficientNet-B0 with a custom classification head.
    Classifier: Dropout → Linear(1280 → num_classes).
    """
    if pretrained:
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    else:
        model = efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features  # 1280

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module):
    """Freeze all feature layers; only classifier is trainable."""
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_all(model: nn.Module):
    """Unfreeze every parameter for full fine-tuning."""
    for p in model.parameters():
        p.requires_grad = True


def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(model: nn.Module, path: Path,
                    class_to_idx: Dict, idx_to_class: Dict,
                    extra: Dict[str, Any] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "target_classes": config.TARGET_CLASSES,
        "num_classes": config.NUM_CLASSES,
        "image_size": config.IMAGE_SIZE,
        "norm_mean": config.NORM_MEAN,
        "norm_std": config.NORM_STD,
        "model_name": config.MODEL_NAME,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device = config.DEVICE):
    """Load checkpoint → (model in eval mode, metadata dict)."""
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Run 'python train.py' first."
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    num_cls = ckpt.get("num_classes", len(ckpt.get("target_classes", config.TARGET_CLASSES)))

    model = build_model(num_classes=num_cls, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt
