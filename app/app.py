"""
Real-time ISL Word-to-Text webcam application.

• Full-frame processing (no hand cropping).
• Optional MediaPipe hand-presence detection (graceful fallback).
• Top-3 predictions displayed.
• Temporal smoothing: 8 stable consecutive frames to commit a word.
• Sentence building with clear/reset controls.
"""

import sys
import time
from pathlib import Path
from typing import List, Optional

# Ensure src/ directory is accessible for imports
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image

import config
from model import load_checkpoint

# ──────────────────────────────────────────────────────────────────────────────
# Optional MediaPipe hand detector (graceful fallback)
# ──────────────────────────────────────────────────────────────────────────────

_mp_hands_module = None
try:
    from mediapipe.python.solutions import hands as _mh
    _mp_hands_module = _mh
except (ImportError, AttributeError):
    try:
        import mediapipe.solutions.hands as _mh
        _mp_hands_module = _mh
    except (ImportError, AttributeError):
        _mp_hands_module = None

if _mp_hands_module is None:
    print("[INFO] MediaPipe Hands not available — hand detection disabled. "
          "Predictions will run on every frame; confidence thresholding will filter noise.")


# ──────────────────────────────────────────────────────────────────────────────
# Application
# ──────────────────────────────────────────────────────────────────────────────

class ISLWebcamApp:
    def __init__(self):
        self.device = config.DEVICE

        # Load model
        print("Loading model...")
        self.model, ckpt = load_checkpoint(config.BEST_MODEL_PATH, self.device)
        self.idx_to_class = ckpt.get("idx_to_class",
                                     {i: c for i, c in enumerate(config.TARGET_CLASSES)})
        self.classes = ckpt.get("target_classes", config.TARGET_CLASSES)
        print(f"Model ready — classes: {self.classes}  device: {self.device}")

        # MediaPipe (optional)
        self.hand_detector = None
        if _mp_hands_module is not None:
            try:
                self.hand_detector = _mp_hands_module.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=config.MP_MIN_DETECTION_CONFIDENCE,
                )
            except Exception as e:
                print(f"[WARN] MediaPipe init failed: {e}")
                self.hand_detector = None

        # Preprocessing — must match training exactly
        self.transform = T.Compose([
            T.Resize(config.IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
        ])

        # State
        self.sentence: List[str] = []
        self.candidate: Optional[str] = None
        self.candidate_count: int = 0
        self.last_committed: Optional[str] = None
        self.hand_gone_since_commit: bool = True
        self.no_hand_frames: int = 0

    # ──────────────────────────────────────────────────────────────────────
    def detect_hand(self, rgb: np.ndarray) -> bool:
        """Returns True if a hand is detected (or if detector unavailable)."""
        if self.hand_detector is None:
            return True  # assume hand present, rely on confidence
        results = self.hand_detector.process(rgb)
        return bool(results and results.multi_hand_landmarks)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, bgr_frame: np.ndarray):
        """Returns (top1_label, top1_conf, top3 list of (label, conf))."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = self.transform(pil).unsqueeze(0).to(self.device)

        use_amp = self.device.type == "cuda"
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = self.model(tensor)

        probs = torch.softmax(logits, dim=1)[0]
        top3_vals, top3_idx = probs.topk(min(3, len(self.classes)))

        top3 = [(self.idx_to_class[top3_idx[i].item()], top3_vals[i].item())
                for i in range(len(top3_idx))]

        return top3[0][0], top3[0][1], top3

    # ──────────────────────────────────────────────────────────────────────
    def update_state(self, hand_present: bool, label: str, conf: float) -> str:
        """Temporal smoothing + sentence building. Returns display status."""
        if not hand_present:
            self.no_hand_frames += 1
            if self.no_hand_frames >= 5:
                self.hand_gone_since_commit = True
            self.candidate = None
            self.candidate_count = 0
            return "No hand detected"

        self.no_hand_frames = 0

        if conf < config.CONFIDENCE_THRESHOLD:
            self.candidate = None
            self.candidate_count = 0
            return f"Low confidence ({conf*100:.0f}%)"

        # Track consecutive stable predictions
        if label == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = label
            self.candidate_count = 1

        # Commit word after enough stable frames
        if self.candidate_count >= config.STABLE_FRAMES_REQUIRED:
            new_word = (label != self.last_committed)
            can_repeat = (label == self.last_committed and self.hand_gone_since_commit)
            if new_word or can_repeat:
                self.sentence.append(label)
                self.last_committed = label
                self.hand_gone_since_commit = False
                print(f"[WORD] {label} ({conf*100:.1f}%) → {' '.join(self.sentence)}")
            self.candidate_count = 0

        return label

    # ──────────────────────────────────────────────────────────────────────
    def draw_hud(self, frame, status, conf, top3, hand_ok, fps):
        h, w = frame.shape[:2]

        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 110), (20, 20, 20), -1)
        cv2.rectangle(overlay, (0, h - 70), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 100, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # Hand status
        hand_text = "Hand: YES" if hand_ok else "Hand: NO"
        hand_color = (0, 255, 0) if hand_ok else (0, 0, 255)
        cv2.putText(frame, hand_text, (w - 130, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, hand_color, 1)

        # Prediction
        if status == "No hand detected":
            cv2.putText(frame, "No hand detected", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif status.startswith("Low"):
            cv2.putText(frame, status, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        else:
            cv2.putText(frame, f"{status} ({conf*100:.1f}%)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Top-3
        if top3:
            for i, (lbl, c) in enumerate(top3):
                cv2.putText(frame, f"{i+1}. {lbl}: {c*100:.1f}%", (15, 60 + i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Stability bar
        ratio = min(self.candidate_count / config.STABLE_FRAMES_REQUIRED, 1.0)
        bar_w = 160
        cv2.rectangle(frame, (15, 100), (15 + bar_w, 108), (80, 80, 80), 1)
        if ratio > 0:
            cv2.rectangle(frame, (15, 100), (15 + int(bar_w * ratio), 108), (0, 255, 128), -1)

        # Sentence output
        sentence_str = " ".join(self.sentence) if self.sentence else "<sign a word>"
        color = (255, 255, 255) if self.sentence else (120, 120, 120)
        cv2.putText(frame, "OUTPUT:", (15, h - 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(frame, sentence_str, (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Controls
        cv2.putText(frame, "[C]lear  [D]elete  [K]save  [Q]uit",
                    (w - 330, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # ──────────────────────────────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(config.WEBCAM_INDEX)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open webcam {config.WEBCAM_INDEX}")
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        win = "ISL Word-to-Text"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        print("\n" + "=" * 50)
        print("  WEBCAM ACTIVE")
        print("  C=Clear  D=Delete  K=Save  Q/ESC=Quit")
        print("=" * 50 + "\n")

        prev_t = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02); continue

                frame = cv2.flip(frame, 1)
                now = time.time()
                fps = 1.0 / max(now - prev_t, 1e-5)
                prev_t = now

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_ok = self.detect_hand(rgb)

                if hand_ok:
                    label, conf, top3 = self.predict(frame)
                else:
                    label, conf, top3 = "", 0.0, []

                status = self.update_state(hand_ok, label, conf)
                self.draw_hud(frame, status, conf, top3, hand_ok, fps)
                cv2.imshow(win, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                elif key in (ord("c"), ord("C")):
                    self.sentence.clear()
                    self.last_committed = None
                    self.candidate = None
                    self.candidate_count = 0
                    print("[CLEAR]")
                elif key in (ord("d"), ord("D"), 8):
                    if self.sentence:
                        removed = self.sentence.pop()
                        self.last_committed = self.sentence[-1] if self.sentence else None
                        print(f"[DELETE] {removed}")
                elif key in (ord("k"), ord("K")):
                    text = " ".join(self.sentence)
                    out = config.RESULTS_DIR / "recognized_output.txt"
                    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                    out.write_text(text + "\n", encoding="utf-8")
                    print(f"[SAVED] {out}: '{text}'")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.hand_detector:
                self.hand_detector.close()
            print("Webcam closed.")


if __name__ == "__main__":
    ISLWebcamApp().run()
