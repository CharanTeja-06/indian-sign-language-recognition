"""
Dataset loading with stratified 80:20 train/val split.
Every user contributes images to BOTH train and val sets.
No participant-level isolation — maximizes learning from all signing styles.
"""

from pathlib import Path
from typing import List, Tuple, Dict
import random

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

import config


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in VALID_EXTENSIONS


def find_class_dir(base: Path, class_name: str) -> Path:
    """Case-insensitive directory lookup."""
    for d in base.iterdir():
        if d.is_dir() and d.name.strip().upper() == class_name.upper():
            return d
    raise FileNotFoundError(
        f"No directory found for class '{class_name}' in {base}. "
        f"Available dirs: {[d.name for d in base.iterdir() if d.is_dir()]}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Collect all samples
# ──────────────────────────────────────────────────────────────────────────────

def collect_all_samples() -> Tuple[List[Tuple[Path, int]], Dict[str, int], Dict[int, str]]:
    """
    Scans dataset directories and returns:
      - samples: list of (image_path, class_index)
      - class_to_idx mapping
      - idx_to_class mapping
    """
    class_to_idx = {name: idx for idx, name in enumerate(config.TARGET_CLASSES)}
    idx_to_class = {idx: name for idx, name in enumerate(config.TARGET_CLASSES)}

    samples = []
    for cls_name in config.TARGET_CLASSES:
        cls_dir = find_class_dir(config.DATA_DIR, cls_name)
        label = class_to_idx[cls_name]

        # Walk through user sub-directories
        user_dirs = sorted([d for d in cls_dir.iterdir() if d.is_dir()])
        if not user_dirs:
            raise RuntimeError(f"No user sub-folders inside {cls_dir}")

        for user_dir in user_dirs:
            for img_path in sorted(user_dir.iterdir()):
                if img_path.is_file() and is_image_file(img_path):
                    samples.append((img_path, label))

    return samples, class_to_idx, idx_to_class


# ──────────────────────────────────────────────────────────────────────────────
# Stratified split
# ──────────────────────────────────────────────────────────────────────────────

def split_dataset(samples: List[Tuple[Path, int]]):
    """
    Stratified 80/20 split.  Images from every user appear in both splits.
    """
    paths  = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels,
        test_size=config.VAL_RATIO,
        random_state=config.RANDOM_SEED,
        stratify=labels,
    )

    train_samples = list(zip(train_paths, train_labels))
    val_samples   = list(zip(val_paths,   val_labels))
    return train_samples, val_samples


# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────

def get_train_transform() -> T.Compose:
    return T.Compose([
        T.Resize(config.IMAGE_SIZE),
        T.RandomRotation(degrees=15),
        T.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.90, 1.10)),
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.04),
        T.RandomPerspective(distortion_scale=0.15, p=0.3),
        T.RandomGrayscale(p=0.05),
        # NO horizontal flip — flipping changes sign meaning
        T.ToTensor(),
        T.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ])


def get_val_transform() -> T.Compose:
    return T.Compose([
        T.Resize(config.IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ISLDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)
        return img, label


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def create_data_loaders(batch_size: int = config.BATCH_SIZE):
    """Returns train_loader, val_loader, class_to_idx, idx_to_class."""
    samples, class_to_idx, idx_to_class = collect_all_samples()
    train_samples, val_samples = split_dataset(samples)

    print(f"Dataset split: {len(train_samples)} train / {len(val_samples)} val")
    for cls_name in config.TARGET_CLASSES:
        idx = class_to_idx[cls_name]
        t_count = sum(1 for _, l in train_samples if l == idx)
        v_count = sum(1 for _, l in val_samples   if l == idx)
        print(f"  {cls_name:8s}: train={t_count:4d}  val={v_count:4d}")

    train_ds = ISLDataset(train_samples, transform=get_train_transform())
    val_ds   = ISLDataset(val_samples,   transform=get_val_transform())

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=pin,
    )

    return train_loader, val_loader, class_to_idx, idx_to_class
