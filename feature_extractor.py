import cv2
import numpy as np
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops

from PIL import Image

def extract_features(image_path_or_nparray, target_size=(128, 128)):
    """
    Extracts classical machine learning features (color, texture, shape) from an image.
    Accepts a file path, a PIL Image object, or a pre-loaded numpy image array (BGR format).
    Returns a 1D numpy array containing the feature vector, and a dictionary of features for debugging.
    """
    # Load image if file path is provided
    if isinstance(image_path_or_nparray, str):
        img = cv2.imread(image_path_or_nparray)
        if img is None:
            raise ValueError(f"Could not load image from path: {image_path_or_nparray}")
    elif isinstance(image_path_or_nparray, Image.Image):
        # Convert PIL to BGR numpy array
        img = cv2.cvtColor(np.array(image_path_or_nparray), cv2.COLOR_RGB2BGR)
    else:
        img = image_path_or_nparray.copy()

    # 1. Image Enhancement & Preprocessing
    # Resize to standardized dimensions
    img_resized = cv2.resize(img, target_size)
    
    # Apply Median Blur to reduce hair noise while preserving edges
    img_blur = cv2.medianBlur(img_resized, 3)
    
    # 2. Color Feature Extraction (RGB and HSV color spaces)
    # Convert BGR to RGB and HSV
    img_rgb = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    
    color_features = []
    feature_dict = {}
    
    # Extract mean, standard deviation, and skewness for RGB channels
    for i, name in enumerate(['R', 'G', 'B']):
        channel = img_rgb[:, :, i]
        mean_val = np.mean(channel)
        std_val = np.std(channel)
        skew_val = skew(channel.flatten())
        color_features.extend([mean_val, std_val, skew_val])
        feature_dict[f'color_rgb_{name}_mean'] = mean_val
        feature_dict[f'color_rgb_{name}_std'] = std_val
        feature_dict[f'color_rgb_{name}_skew'] = skew_val

    # Extract mean, standard deviation, and skewness for HSV channels
    for i, name in enumerate(['H', 'S', 'V']):
        channel = img_hsv[:, :, i]
        mean_val = np.mean(channel)
        std_val = np.std(channel)
        skew_val = skew(channel.flatten())
        color_features.extend([mean_val, std_val, skew_val])
        feature_dict[f'color_hsv_{name}_mean'] = mean_val
        feature_dict[f'color_hsv_{name}_std'] = std_val
        feature_dict[f'color_hsv_{name}_skew'] = skew_val

    # 3. Texture Feature Extraction (GLCM - Gray-Level Co-occurrence Matrix)
    # Convert to grayscale
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    
    # Compute GLCM with distance=1 and distance=3, angles=0, 45, 90, 135 degrees
    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    texture_features = []
    properties = ['contrast', 'correlation', 'energy', 'homogeneity']
    
    for dist_idx, dist in enumerate(distances):
        for prop in properties:
            # Calculate properties and average across angles
            values = graycoprops(glcm, prop)[dist_idx, :]
            mean_val = np.mean(values)
            texture_features.append(mean_val)
            feature_dict[f'texture_glcm_d{dist}_{prop}'] = mean_val

    # 4. Shape & Morphological Feature Extraction
    # Convert to grayscale and apply thresholding to segment the lesion
    # Use Otsu's thresholding on a blurred version of grayscale
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize default shape features
    area = 0.0
    perimeter = 0.0
    circularity = 0.0
    solidity = 0.0
    eccentricity = 0.0
    
    if contours:
        # Find the largest contour (representing the primary lesion)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Avoid division by zero
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
        
        # Solidity
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            
        # Eccentricity / Aspect ratio via fitEllipse
        if len(largest_contour) >= 5:
            (x, y), (MA, ma), angle = cv2.fitEllipse(largest_contour)
            if ma > 0:
                eccentricity = np.sqrt(1 - (MA / ma) ** 2) if MA <= ma else np.sqrt(1 - (ma / MA) ** 2)
        else:
            # Fallback to bounding box aspect ratio
            x, y, w, h = cv2.boundingRect(largest_contour)
            if max(w, h) > 0:
                eccentricity = min(w, h) / max(w, h)

    shape_features = [area, perimeter, circularity, solidity, eccentricity]
    feature_dict['shape_area'] = area
    feature_dict['shape_perimeter'] = perimeter
    feature_dict['shape_circularity'] = circularity
    feature_dict['shape_solidity'] = solidity
    feature_dict['shape_eccentricity'] = eccentricity
    
    # Combine all feature arrays
    all_features = np.array(color_features + texture_features + shape_features, dtype=np.float32)
    return all_features, feature_dict
