"""
MediaPipe Hand Detection & Cropping Module.

Provides consistent hand detection, bounding box estimation, padding, and ROI cropping.
This module is strictly shared between dataset preprocessing and real-time webcam inference
to guarantee that input distributions match between training and deployment.
"""

from typing import Tuple, Optional, Union
import numpy as np
from PIL import Image
import cv2

# Safe initialization for MediaPipe Hands
# In MediaPipe >= 1.0 (such as on Python 3.13), 'solutions' is removed and replaced by 'tasks'.
_MP_AVAILABLE = False
mp_hands = None

try:
    from mediapipe.python.solutions import hands as _mp_hands
    mp_hands = _mp_hands
    _MP_AVAILABLE = True
except (ImportError, AttributeError):
    try:
        import mediapipe.solutions.hands as _mp_hands
        mp_hands = _mp_hands
        _MP_AVAILABLE = True
    except (ImportError, AttributeError):
        mp_hands = None
        _MP_AVAILABLE = False


class HandCropPipeline:
    """
    Encapsulates MediaPipe Hand detection and square bounding-box cropping with padding.
    """

    def __init__(self,
                 min_detection_confidence: float = 0.6,
                 min_tracking_confidence: float = 0.5,
                 padding_ratio: float = 0.25):
        """
        Args:
            min_detection_confidence: MediaPipe minimum detection confidence.
            min_tracking_confidence: MediaPipe minimum tracking confidence.
            padding_ratio: Margin to add around the detected hand bounding box (e.g. 0.25 = 25%).
        """
        self.padding_ratio = padding_ratio
        self.mp_available = _MP_AVAILABLE and (mp_hands is not None)

        if self.mp_available:
            self.mp_hands = mp_hands
            self.detector = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        else:
            self.detector = None

    def process(self,
                image: Union[np.ndarray, Image.Image],
                target_size: Tuple[int, int] = (224, 224)
                ) -> Tuple[np.ndarray, bool, Optional[Tuple[int, int, int, int]]]:
        """
        Detects hand and crops the region of interest.

        Args:
            image: Input image as RGB numpy array (H, W, 3) or PIL Image.
            target_size: Output resolution (width, height).

        Returns:
            cropped_image: RGB numpy array resized to target_size.
            hand_detected: Boolean indicating if a hand was detected.
            bbox: (x1, y1, x2, y2) bounding box in original image coords, or None.
        """
        # Convert PIL Image to RGB numpy array if necessary
        if isinstance(image, Image.Image):
            rgb_np = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            rgb_np = image.copy()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        h, w = rgb_np.shape[:2]

        if not self.mp_available or self.detector is None:
            # Fallback when mediapipe is not available: center-resize image
            fallback_resized = cv2.resize(rgb_np, target_size, interpolation=cv2.INTER_AREA)
            return fallback_resized, False, None

        # MediaPipe expects RGB image
        results = self.detector.process(rgb_np)

        if not results.multi_hand_landmarks:
            # No hand detected: fallback to resizing full frame
            fallback_resized = cv2.resize(rgb_np, target_size, interpolation=cv2.INTER_AREA)
            return fallback_resized, False, None

        # Take first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        bw = max_x - min_x
        bh = max_y - min_y

        # Make bounding box square to avoid distortion
        box_size = max(bw, bh)
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0

        # Add padding margin
        padded_size = box_size * (1.0 + self.padding_ratio)

        # Compute integer coordinates clamped to image borders
        x1 = max(0, int(cx - padded_size / 2.0))
        y1 = max(0, int(cy - padded_size / 2.0))
        x2 = min(w, int(cx + padded_size / 2.0))
        y2 = min(h, int(cy + padded_size / 2.0))

        # Check for valid crop region
        if x2 - x1 < 10 or y2 - y1 < 10:
            fallback_resized = cv2.resize(rgb_np, target_size, interpolation=cv2.INTER_AREA)
            return fallback_resized, False, None

        crop = rgb_np[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)

        return crop_resized, True, (x1, y1, x2, y2)

    def close(self):
        """Release MediaPipe resources."""
        if self.detector:
            self.detector.close()


# Module-level convenience instance for quick imports
_shared_pipeline = None


def get_shared_pipeline() -> HandCropPipeline:
    global _shared_pipeline
    if _shared_pipeline is None:
        _shared_pipeline = HandCropPipeline()
    return _shared_pipeline


def crop_hand(image: Union[np.ndarray, Image.Image],
              target_size: Tuple[int, int] = (224, 224)
              ) -> Tuple[np.ndarray, bool, Optional[Tuple[int, int, int, int]]]:
    """
    Standard function to crop hand from image.
    Used by both dataset loaders and live webcam inference.
    """
    pipeline = get_shared_pipeline()
    return pipeline.process(image, target_size=target_size)
