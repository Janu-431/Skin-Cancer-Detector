"""
utils/preprocessing.py
=======================
Enhanced image preprocessing utilities for the Skin Cancer Detection app.

Provides three preprocessing pipelines:
  1. preprocess_for_display  — mild denoise + resize for UI rendering
  2. preprocess_for_gradcam  — MobileNetV2-compatible 224×224 normalization
  3. preprocess_for_rf       — wrapper that calls the existing feature extractor

All functions accept either a PIL Image or a NumPy BGR array and return
a PIL Image (or the feature vector in the RF case).
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter

# ─── Constants ─────────────────────────────────────────────────────────────────
GRADCAM_INPUT_SIZE = (224, 224)   # MobileNetV2 / EfficientNetB0 standard input
DISPLAY_MAX_SIDE   = 512          # Max side for display thumbnails


# ─── Internal helper ───────────────────────────────────────────────────────────

def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to an OpenCV BGR uint8 array."""
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR uint8 array to a PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ─── Pipeline 1: Display Preprocessing ────────────────────────────────────────

def preprocess_for_display(pil_img: Image.Image, max_side: int = DISPLAY_MAX_SIDE) -> Image.Image:
    """
    Prepare an image for clean UI display.

    Steps:
      - Convert to RGB (handles RGBA, grayscale, etc.)
      - Resize so the longest side ≤ max_side (aspect-ratio preserved)
      - Apply a mild median blur via PIL for noise reduction
      - Normalize pixel values remain uint8 for direct display

    Args:
        pil_img  : Input PIL image (any mode).
        max_side : Maximum pixel length of longest side after resize.

    Returns:
        PIL.Image in RGB mode, ready for `st.image()`.
    """
    img = pil_img.convert("RGB")

    # Aspect-ratio-preserving resize
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # Mild noise reduction (radius=1 keeps edges sharp)
    img = img.filter(ImageFilter.MedianFilter(size=3))

    return img


# ─── Pipeline 2: Grad-CAM / CNN Preprocessing ─────────────────────────────────

def preprocess_for_gradcam(pil_img: Image.Image) -> np.ndarray:
    """
    Prepare a PIL image for MobileNetV2 / EfficientNetB0 inference + Grad-CAM.

    Steps:
      1. Convert to RGB
      2. Apply OpenCV median blur (hair / noise removal)
      3. CLAHE contrast enhancement on L-channel (improves lesion visibility)
      4. Resize to 224×224
      5. Normalize to [0, 1] float32
      6. Expand batch dimension → shape (1, 224, 224, 3)

    Args:
        pil_img : Input PIL image (any mode).

    Returns:
        np.ndarray of shape (1, 224, 224, 3), dtype float32, values in [0, 1].
    """
    # Step 1 – ensure RGB
    img_rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Step 2 – median blur for impulse noise / hair removal
    img_bgr = cv2.medianBlur(img_bgr, 5)

    # Step 3 – CLAHE on L-channel for contrast enhancement
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    img_lab = cv2.merge([l_ch, a_ch, b_ch])
    img_bgr = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

    # Step 4 – resize to MobileNetV2 input size
    img_resized = cv2.resize(img_bgr, GRADCAM_INPUT_SIZE, interpolation=cv2.INTER_AREA)

    # Step 5 – convert back to RGB, normalize to [0, 1]
    img_rgb_norm = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Step 6 – add batch dimension
    return np.expand_dims(img_rgb_norm, axis=0)


# ─── Pipeline 3: RF Feature Preprocessing (wrapper) ───────────────────────────

def preprocess_for_rf(pil_img: Image.Image):
    """
    Wrapper around the existing feature extractor for clarity and modularity.

    Args:
        pil_img : Input PIL image.

    Returns:
        Tuple (feature_vector: np.ndarray, feature_dict: dict) as returned
        by `feature_extractor.extract_features`.
    """
    # Import here to avoid circular imports at module load time
    from feature_extractor import extract_features
    return extract_features(pil_img)


# ─── Utility: Convert numpy array to PNG bytes for download ───────────────────

def pil_to_png_bytes(pil_img: Image.Image) -> bytes:
    """
    Encode a PIL image to PNG bytes suitable for `st.download_button`.

    Args:
        pil_img : PIL image to encode.

    Returns:
        Raw PNG bytes (in-memory, no temp file).
    """
    import io
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG", optimize=False)
    buf.seek(0)
    return buf.read()
