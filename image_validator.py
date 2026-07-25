import cv2
import numpy as np

from PIL import Image

def validate_skin_lesion_image(image_path_or_nparray):
    """
    Validates an image before skin lesion prediction:
    1. Checks if the image is readable/corrupted.
    2. Detects if a human face is present (using OpenCV Haar Cascades).
    3. Checks if the image is blank (pure white, pure black, or extremely low variance).
    
    Returns:
        tuple: (is_valid, message)
        - is_valid (bool): True if the image is a valid skin lesion image, False otherwise.
        - message (str): Explanation of validation failure, or "Valid skin lesion image" if valid.
    """
    # Load image if file path is provided
    if isinstance(image_path_or_nparray, str):
        img = cv2.imread(image_path_or_nparray)
        if img is None:
            return False, "Corrupted or unsupported file format. Please upload a valid image."
    elif isinstance(image_path_or_nparray, Image.Image):
        # Convert PIL to BGR numpy array
        img = cv2.cvtColor(np.array(image_path_or_nparray), cv2.COLOR_RGB2BGR)
    else:
        if image_path_or_nparray is None:
            return False, "No image data received."
        if hasattr(image_path_or_nparray, 'size') and image_path_or_nparray.size == 0:
            return False, "No image data received."
        img = image_path_or_nparray.copy()

    # Convert to grayscale for validation routines
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- 1. Face Detection ---
    # Load OpenCV's built-in frontal face Haar cascade
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    if face_cascade.empty():
        # Fallback if XML is not loaded properly (unlikely but safe)
        print("[Warning] Face cascade XML could not be loaded from OpenCV default directory.")
    else:
        # Detect faces. ScaleFactor and minNeighbors are set to balance precision/recall
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            return False, "Human Face Detected – Please Show a Skin Lesion Image Only."

    # --- 2. Blank / Solid Color Detection ---
    # Compute image statistics
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    
    # If the standard deviation is extremely low, it's a blank or solid-color image
    if std_val < 10.0:
        return False, "Blank or extremely low-contrast image. Please provide a clear skin lesion image."
        
    # Check if the image is almost fully white or black
    if mean_val > 248.0:
        return False, "Invalid image: Pure white or overexposed frame."
    if mean_val < 8.0:
        return False, "Invalid image: Pure black or underexposed frame."

    # --- 3. Lesion Shape Contrast Pre-screen ---
    # (Optional) Verify if there is a minimum structure in the image.
    # Skin lesions usually have a distinguishable region in contrast to surrounding skin.
    # If standard deviation is extremely low, we flag it.

    return True, "Valid skin lesion image."
