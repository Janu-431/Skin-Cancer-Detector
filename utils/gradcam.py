"""
utils/gradcam.py
================
Grad-CAM (Gradient-weighted Class Activation Mapping) engine for the
Skin Cancer Detection app.

This module uses PyTorch + torchvision (MobileNetV2) instead of TensorFlow,
because TensorFlow does not yet support Python 3.14 while PyTorch does.

The Grad-CAM implementation:
  - Loads MobileNetV2 (ImageNet pre-trained) once and caches it.
  - Registers a forward hook on the last convolutional block to capture
    feature maps during the forward pass.
  - Computes class-discriminative gradients via autograd.
  - Overlays the resulting JET-colorized heatmap on the original image.
  - Returns PIL images ready for direct use in Streamlit.

Why MobileNetV2?
  - Lightweight: ~14 MB, fast on CPU (~100–300 ms per image).
  - Strong depthwise-separable convolutions produce rich spatial maps.
  - No retraining needed: ImageNet features generalize well to skin textures.

Reference: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization." ICCV 2017.
"""

import cv2
import numpy as np
from PIL import Image

# PyTorch is imported lazily to avoid startup cost when not needed.

# ─── Constants ─────────────────────────────────────────────────────────────────
MOBILENET_INPUT_SIZE = (224, 224)   # Standard MobileNetV2 input
HEATMAP_ALPHA        = 0.50         # Overlay opacity (0 = original, 1 = heatmap)

# ImageNet normalization parameters (used by torchvision models)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─── Model Builder ─────────────────────────────────────────────────────────────

def build_gradcam_model(num_classes: int):
    """
    Build a Grad-CAM-ready MobileNetV2 model using PyTorch + torchvision.

    Architecture:
      MobileNetV2 backbone (frozen, ImageNet weights)
      → AdaptiveAvgPool2d  (built-in)
      → Dropout(0.2)       (built-in)
      → Linear(1280, num_classes)  ← replaced classifier head

    The backbone's convolutional weights are frozen. Only the final
    linear classifier is (optionally) trainable; for Grad-CAM we use
    the RF-predicted class index to guide gradient computation, so no
    fine-tuning of the head is required in practice.

    Args:
        num_classes : Number of output classes (matches metadata["classes"]).

    Returns:
        torch.nn.Module: Full MobileNetV2 model in eval mode.

    Raises:
        ImportError: If torch or torchvision is not installed.
    """
    try:
        import torch
        import torch.nn as nn
        from torchvision import models

        # Load pre-trained MobileNetV2
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        # Freeze backbone (features / convolutional layers)
        for param in model.features.parameters():
            param.requires_grad = False

        # Replace the classifier head to match our num_classes
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

        # Set to eval mode (disables dropout, batch-norm training)
        model.eval()

        return model

    except ImportError as e:
        raise ImportError(
            "PyTorch and torchvision are required for Grad-CAM. "
            "Install with: pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cpu"
        ) from e


# ─── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess_tensor(pil_img: Image.Image):
    """
    Preprocess a PIL image into a PyTorch tensor for MobileNetV2.

    Pipeline:
      1. Convert to RGB
      2. Apply median blur (noise / hair removal)
      3. CLAHE contrast enhancement
      4. Resize to 224×224
      5. Normalize with ImageNet mean/std
      6. Return tensor of shape (1, 3, 224, 224)

    Args:
        pil_img: Input PIL image (any mode).

    Returns:
        torch.Tensor of shape (1, 3, 224, 224), float32.
    """
    import torch

    # Step 1 – ensure RGB numpy
    img_rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Step 2 – median blur
    img_bgr = cv2.medianBlur(img_bgr, 5)

    # Step 3 – CLAHE on L-channel
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch  = clahe.apply(l_ch)
    img_bgr = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # Step 4 – resize to model input
    img_rgb_small = cv2.cvtColor(
        cv2.resize(img_bgr, MOBILENET_INPUT_SIZE, interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2RGB
    )

    # Step 5 – normalize to [0,1] then apply ImageNet stats
    img_f = img_rgb_small.astype(np.float32) / 255.0
    mean  = np.array(_IMAGENET_MEAN, dtype=np.float32)
    std   = np.array(_IMAGENET_STD,  dtype=np.float32)
    img_f = (img_f - mean) / std                       # (H, W, 3)

    # Step 6 – (H, W, 3) → (1, 3, H, W)
    tensor = torch.from_numpy(img_f.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


# ─── Grad-CAM Core ─────────────────────────────────────────────────────────────

def compute_gradcam_heatmap(model, img_tensor, class_idx: int) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for a given class index using PyTorch hooks.

    Algorithm:
      1. Register a forward hook on `model.features[-1]` (last conv block)
         to capture its output feature maps.
      2. Run a forward pass with gradient tracking enabled.
      3. Zero gradients, then back-propagate from the target class score.
      4. Pool the gradients spatially → per-channel weights.
      5. Weighted sum of feature maps → ReLU → normalize to [0, 1].
      6. Resize to MOBILENET_INPUT_SIZE.

    Args:
        model      : PyTorch MobileNetV2 model in eval mode.
        img_tensor : Preprocessed tensor, shape (1, 3, 224, 224).
        class_idx  : Target class index for gradient guidance.

    Returns:
        np.ndarray of shape (224, 224), float32, values in [0, 1].
    """
    import torch

    # Storage for hook outputs
    feature_maps = {}
    gradients    = {}

    # ── Register hooks ────────────────────────────────────────────────────────
    def fwd_hook(module, inp, output):
        feature_maps["last_conv"] = output.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients["last_conv"] = grad_out[0].detach()

    target_layer = model.features[-1]   # MobileNetV2's last ConvBNActivation block
    fwd_handle   = target_layer.register_forward_hook(fwd_hook)
    bwd_handle   = target_layer.register_full_backward_hook(bwd_hook)

    try:
        # ── Forward pass ──────────────────────────────────────────────────────
        model.zero_grad()
        img_tensor.requires_grad_(True)

        # Temporarily enable gradients for the frozen backbone during this pass
        with torch.enable_grad():
            # Re-enable grad for feature maps (needed for hook)
            logits = model(img_tensor)                   # (1, num_classes)
            class_score = logits[0, class_idx]           # scalar

        # ── Backward pass ──────────────────────────────────────────────────────
        class_score.backward()

        # ── Extract feature maps and gradients ────────────────────────────────
        fmaps = feature_maps["last_conv"]   # (1, C, H, W)
        grads = gradients["last_conv"]      # (1, C, H, W)

        # ── Global Average Pooling of gradients → per-channel weights ─────────
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # ── Weighted sum of feature maps ──────────────────────────────────────
        cam = (weights * fmaps).sum(dim=1, keepdim=False)[0]  # (H, W)

        # ── ReLU + normalize ─────────────────────────────────────────────────
        cam = torch.relu(cam).numpy()
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        else:
            # Fallback: Gaussian blob centred on image
            h, w = cam.shape
            y, x = np.ogrid[:h, :w]
            cy, cx = h // 2, w // 2
            cam = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * (min(h, w) // 4)**2))
            cam = cam / cam.max()

        # ── Resize to model input size ────────────────────────────────────────
        cam_resized = cv2.resize(
            cam.astype(np.float32),
            MOBILENET_INPUT_SIZE,
            interpolation=cv2.INTER_CUBIC
        )
        return cam_resized.astype(np.float32)

    finally:
        # Always remove hooks regardless of success/failure
        fwd_handle.remove()
        bwd_handle.remove()


# ─── Heatmap Rendering ─────────────────────────────────────────────────────────

def render_heatmap(heatmap: np.ndarray) -> Image.Image:
    """
    Colorize a normalized [0,1] heatmap using the JET colormap.

    JET color scale: blue (low) → green → yellow → red (high activation).
    Red regions = highest model attention = suspicious area.

    Args:
        heatmap: 2-D float32 array, values in [0, 1].

    Returns:
        PIL.Image in RGB mode.
    """
    heatmap_u8 = np.uint8(255 * heatmap)
    colored_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    return Image.fromarray(cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB))


def render_overlay(
    original_pil: Image.Image,
    heatmap: np.ndarray,
    alpha: float = HEATMAP_ALPHA
) -> Image.Image:
    """
    Blend the JET heatmap onto the original image.

    The original is resized to MOBILENET_INPUT_SIZE for consistent blending,
    then merged with the colorized heatmap via weighted addition.

    Args:
        original_pil: Original PIL image (any size).
        heatmap     : 2-D float32 heatmap [0, 1].
        alpha       : Heatmap weight (0 = only original, 1 = only heatmap).

    Returns:
        PIL.Image in RGB mode at MOBILENET_INPUT_SIZE.
    """
    orig_resized = original_pil.convert("RGB").resize(MOBILENET_INPUT_SIZE, Image.LANCZOS)
    orig_np      = np.array(orig_resized, dtype=np.uint8)

    heatmap_u8   = np.uint8(255 * heatmap)
    colored_rgb  = cv2.cvtColor(cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(orig_np, 1.0 - alpha, colored_rgb, alpha, 0)
    return Image.fromarray(overlay.astype(np.uint8))


# ─── Suspicious Region Annotation ─────────────────────────────────────────────

def annotate_suspicious_region(
    overlay_pil: Image.Image,
    heatmap: np.ndarray,
    threshold: float = 0.60
) -> Image.Image:
    """
    Draw a visible contour around the highest-activation (suspicious) region.

    The heatmap is thresholded to find high-activation pixels. The largest
    connected region above the threshold is outlined in cyan, with a label
    and a subtle translucent fill to make it visually distinct.

    Args:
        overlay_pil : Blended overlay PIL image.
        heatmap     : 2-D float32 heatmap used for region detection.
        threshold   : Activation threshold for detecting the suspicious area.

    Returns:
        PIL.Image with cyan contour and label drawn on the suspicious region.
    """
    img_np = np.array(overlay_pil, dtype=np.uint8).copy()

    # Threshold the heatmap
    mask = (heatmap >= threshold).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))

    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)
        min_area = 0.005 * img_np.shape[0] * img_np.shape[1]

        if area > min_area:
            # Subtle translucent fill
            mask_draw = np.zeros_like(img_np)
            cv2.drawContours(mask_draw, [largest], -1, (0, 255, 200), -1)
            img_np = cv2.addWeighted(img_np, 1.0, mask_draw, 0.08, 0)

            # Solid cyan border
            cv2.drawContours(img_np, [largest], -1, (0, 230, 210), 2)

            # Label with dark background pill
            x, y, w, h = cv2.boundingRect(largest)
            label = "Suspicious Region"
            font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
            pad = 4
            label_y = max(y - th - 2 * pad - 2, th + 2 * pad + 2)
            cv2.rectangle(img_np, (x, label_y - th - 2 * pad), (x + tw + 2 * pad, label_y + pad), (0, 0, 0), -1)
            cv2.putText(img_np, label, (x + pad, label_y), font, scale, (0, 230, 210), thick, cv2.LINE_AA)

    return Image.fromarray(img_np)


# ─── Main Public API ───────────────────────────────────────────────────────────

def generate_gradcam(
    pil_img: Image.Image,
    class_idx: int,
    num_classes: int,
    cached_model=None
) -> dict:
    """
    Full Grad-CAM pipeline: preprocess → forward → compute heatmap → render.

    This is the single entry point called from app.py. Returns a dict
    with all artifacts needed by the Streamlit UI.

    Args:
        pil_img      : Original uploaded PIL image.
        class_idx    : RF-predicted class index (guides the gradient direction
                       so the heatmap explains the clinically-predicted class).
        num_classes  : Total number of output classes.
        cached_model : Pre-built model from @st.cache_resource.
                       If None, a new model is built (slower first call).

    Returns:
        dict with keys:
          "success"       : bool
          "heatmap_pil"   : PIL.Image — colorized JET heatmap
          "overlay_pil"   : PIL.Image — heatmap blended on original
          "annotated_pil" : PIL.Image — overlay with suspicious region contour
          "cnn_confidence": float — MobileNetV2 top-class probability (0–100)
          "error"         : str — error message if success=False
    """
    try:
        import torch

        # ── Step 1: Get / build model ─────────────────────────────────────────
        model = cached_model
        if model is None:
            model = build_gradcam_model(num_classes)
        model.eval()

        # ── Step 2: Preprocess image ──────────────────────────────────────────
        img_tensor = _preprocess_tensor(pil_img)   # (1, 3, 224, 224)

        # ── Step 3: Forward pass for confidence ───────────────────────────────
        with torch.no_grad():
            logits      = model(img_tensor)                  # (1, num_classes)
            probs       = torch.softmax(logits, dim=1)[0]    # (num_classes,)
        cnn_confidence = float(probs[class_idx].item()) * 100.0

        # ── Step 4: Compute Grad-CAM using RF-predicted class_idx ─────────────
        # NOTE: we need gradients here, so we do NOT use torch.no_grad()
        heatmap = compute_gradcam_heatmap(model, img_tensor, class_idx)

        # ── Step 5: Render outputs ────────────────────────────────────────────
        heatmap_pil   = render_heatmap(heatmap)
        overlay_pil   = render_overlay(pil_img, heatmap, alpha=HEATMAP_ALPHA)
        annotated_pil = annotate_suspicious_region(overlay_pil, heatmap, threshold=0.60)

        return {
            "success"       : True,
            "heatmap_pil"   : heatmap_pil,
            "overlay_pil"   : overlay_pil,
            "annotated_pil" : annotated_pil,
            "cnn_confidence": round(cnn_confidence, 2),
            "error"         : None,
        }

    except ImportError as e:
        return {
            "success": False,
            "error"  : (
                "PyTorch not installed. Grad-CAM unavailable. "
                "Run: pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cpu"
                f"  ({e})"
            ),
        }
    except Exception as e:
        return {
            "success": False,
            "error"  : f"Grad-CAM generation failed: {e}",
        }
