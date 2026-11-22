"""
Dataset Audit Script for ISL Word-to-Text Recognition.

Performs a rigorous check on the dataset before any training:
1. Detects actual dataset folder structure and resolves class folders (case-insensitive).
2. Counts images per word and per participant (User).
3. Identifies unique participant IDs across classes and flags discrepancies.
4. Checks for missing folders and corrupted/unreadable images.
5. Inspects image dimensions and aspect ratios.
6. Generates a contact sheet of 12 sample images per class saved to results/audit_contact_sheet.png.
7. Reports class and participant imbalances.
"""

import sys
import random
from pathlib import Path
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt

import config


def find_class_directories(base_dir: Path, target_classes: list) -> dict:
    """
    Locates directories corresponding to target classes in a case-insensitive manner.
    Returns a mapping: {TARGET_NAME: Path_to_folder}
    """
    if not base_dir.exists():
        return {}

    subdirs = [p for p in base_dir.iterdir() if p.is_dir()]
    mapping = {}

    for target in target_classes:
        match = None
        for subdir in subdirs:
            if subdir.name.strip().upper() == target.upper():
                match = subdir
                break
        if match:
            mapping[target] = match

    return mapping


def is_image_file(path: Path) -> bool:
    """Checks whether a file has a recognized image extension."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return path.suffix.lower() in valid_exts


def run_audit():
    print("=" * 70)
    print("       INDIAN SIGN LANGUAGE (ISL) DATASET INTEGRITY AUDIT")
    print("=" * 70)
    print(f"Project Root: {config.PROJECT_ROOT}")
    print(f"Target Classes: {config.TARGET_CLASSES}")
    print(f"Searching dataset in: {config.DATA_DIR.resolve()}\n")

    # Step 1: Detect class folders
    class_dirs = find_class_directories(config.DATA_DIR, config.TARGET_CLASSES)

    missing_classes = set(config.TARGET_CLASSES) - set(class_dirs.keys())
    if missing_classes:
        print(f"[ERROR] Missing class directories for: {sorted(list(missing_classes))}")
        print(f"Available directories in {config.DATA_DIR}: {[d.name for d in config.DATA_DIR.iterdir() if d.is_dir()]}")
        print("\nStopping audit because dataset folder structure is incomplete or unrecognized.")
        sys.exit(1)

    print("[SUCCESS] All 6 target class directories found:")
    for cls_name, cls_path in class_dirs.items():
        print(f"  - {cls_name:8s} -> {cls_path.name}/")
    print()

    # Step 2: Audit participants, image counts, dimensions, and corruptions
    total_images = 0
    class_image_counts = defaultdict(int)
    class_user_counts = defaultdict(lambda: defaultdict(int))
    user_appearances = defaultdict(set)
    dimensions_set = set()
    corrupted_files = []
    class_samples = defaultdict(list)

    for cls_name, cls_path in class_dirs.items():
        # User folders inside class directory
        user_dirs = [p for p in cls_path.iterdir() if p.is_dir()]

        if not user_dirs:
            print(f"[WARNING] No participant/user subdirectories found inside {cls_path.name}/")
            # Fallback check for images directly in class folder
            direct_images = [p for p in cls_path.iterdir() if p.is_file() and is_image_file(p)]
            if direct_images:
                print(f"[ERROR] Found {len(direct_images)} images directly in {cls_path.name}/.")
                print("Dataset must be organized with User folders: <Class>/<User_ID>/<images>.")
                sys.exit(1)

        for user_dir in sorted(user_dirs, key=lambda x: x.name):
            user_id = user_dir.name
            user_appearances[user_id].add(cls_name)

            img_paths = [p for p in user_dir.iterdir() if p.is_file() and is_image_file(p)]
            for img_p in img_paths:
                total_images += 1
                class_image_counts[cls_name] += 1
                class_user_counts[cls_name][user_id] += 1

                # Verify image readability and check dimensions
                try:
                    with Image.open(img_p) as img:
                        img.verify()  # Check for integrity
                    with Image.open(img_p) as img:
                        dimensions_set.add(img.size)  # (width, height)
                    class_samples[cls_name].append(img_p)
                except Exception as e:
                    corrupted_files.append((img_p, str(e)))

    # Step 3: Print Findings
    print("-" * 70)
    print("1. PARTICIPANT CONSISTENCY & DISTRIBUTION")
    print("-" * 70)
    all_users = sorted(list(user_appearances.keys()))
    print(f"Detected {len(all_users)} unique participants: {', '.join(all_users)}")

    common_users = [u for u in all_users if len(user_appearances[u]) == len(config.TARGET_CLASSES)]
    inconsistent_users = [u for u in all_users if len(user_appearances[u]) < len(config.TARGET_CLASSES)]

    print(f"Participants present across ALL 6 classes ({len(common_users)}): {common_users}")
    if inconsistent_users:
        print(f"[NOTE] Participants missing in some classes ({len(inconsistent_users)}):")
        for u in inconsistent_users:
            missing_in = set(config.TARGET_CLASSES) - user_appearances[u]
            print(f"  - '{u}' is missing in: {sorted(list(missing_in))}")
    print()

    print("-" * 70)
    print("2. IMAGE COUNTS PER CLASS & PARTICIPANT")
    print("-" * 70)
    print(f"{'Class':<10} | {'Total':<7} | {'Breakdown by Participant'}")
    print("-" * 70)
    for cls_name in config.TARGET_CLASSES:
        breakdown = ", ".join(f"{uid}: {class_user_counts[cls_name][uid]}" for uid in sorted(class_user_counts[cls_name].keys()))
        print(f"{cls_name:<10} | {class_image_counts[cls_name]:<7} | {breakdown}")
    print(f"\nTotal images across entire dataset: {total_images}")

    # Check class balance
    if class_image_counts:
        counts = list(class_image_counts.values())
        min_c, max_c = min(counts), max(counts)
        ratio = max_c / (min_c if min_c > 0 else 1)
        if ratio > 1.5:
            print(f"[IMBALANCE WARNING] Class ratio max/min is {ratio:.2f}. Class balancing may be advised.")
        else:
            print(f"[OK] Class distribution is reasonably balanced (ratio: {ratio:.2f}).")
    print()

    print("-" * 70)
    print("3. IMAGE INTEGRITY & DIMENSIONS")
    print("-" * 70)
    if corrupted_files:
        print(f"[CRITICAL] Found {len(corrupted_files)} corrupted or unreadable images:")
        for path, err in corrupted_files[:10]:
            print(f"  - {path}: {err}")
        if len(corrupted_files) > 10:
            print(f"  ... and {len(corrupted_files) - 10} more.")
    else:
        print("[OK] No corrupted images found! All tested images read successfully.")

    print(f"Distinct image dimensions (width x height) found ({len(dimensions_set)}):")
    for dims in sorted(list(dimensions_set))[:5]:
        print(f"  - {dims[0]}x{dims[1]} px")
    if len(dimensions_set) > 5:
        print(f"  ... and {len(dimensions_set) - 5} more variations.")
    print()

    # Step 4: Contact Sheet Generation (12 samples per word)
    print("-" * 70)
    print("4. GENERATING CONTACT SHEET (12 SAMPLES PER WORD)")
    print("-" * 70)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    contact_sheet_path = config.RESULTS_DIR / "audit_contact_sheet.png"

    fig, axes = plt.subplots(nrows=len(config.TARGET_CLASSES), ncols=12, figsize=(20, 2.8 * len(config.TARGET_CLASSES)))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    random.seed(config.RANDOM_SEED)
    for row_idx, cls_name in enumerate(config.TARGET_CLASSES):
        samples = class_samples[cls_name]
        chosen = random.sample(samples, min(12, len(samples))) if len(samples) >= 12 else samples
        while len(chosen) < 12 and len(samples) > 0:
            chosen.append(random.choice(samples))

        for col_idx in range(12):
            ax = axes[row_idx, col_idx]
            if col_idx < len(chosen):
                img_path = chosen[col_idx]
                try:
                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "Error", ha="center", va="center")
            ax.axis("off")
            if col_idx == 0:
                ax.set_title(cls_name, fontsize=12, fontweight="bold", loc="left")

    plt.tight_layout()
    fig.savefig(contact_sheet_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Contact sheet saved to: {contact_sheet_path.resolve()}\n")

    print("=" * 70)
    print("                      AUDIT COMPLETE")
    print("=" * 70)
    print("Inspection complete. Please review the output above and the contact sheet.")
    print("When ready, proceed to dataset splitting and model training.")


if __name__ == "__main__":
    run_audit()
