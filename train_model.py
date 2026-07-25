import os
import sys
import json
import joblib
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support

# Import feature extractor
from feature_extractor import extract_features

def generate_mock_images_for_metadata(df, image_dir, limit=200):
    """
    Generates synthetic lesion images matching the metadata Excel sheet for testing.
    This allows running the project immediately without a 2.7GB download.
    """
    print(f"\n[!] Images not found in '{image_dir}'. Generating {limit} synthetic lesion images for testing...")
    os.makedirs(image_dir, exist_ok=True)
    
    # Class-specific visual profiles
    classes_props = {
        "nv": {"color": (40, 60, 90), "regular": True, "noise": 10},            # Nevus: uniform, round
        "mel": {"color": (20, 30, 50), "regular": False, "noise": 30},         # Melanoma: dark, irregular
        "bkl": {"color": (30, 80, 110), "regular": False, "noise": 20},        # Benign keratosis: waxy
        "bcc": {"color": (25, 45, 75), "regular": False, "noise": 25},         # Basal cell: pearly/brown
        "akiec": {"color": (50, 50, 120), "regular": False, "noise": 20},      # Actinic: scaly/reddish
        "vasc": {"color": (15, 20, 150), "regular": True, "noise": 15},        # Vascular: red/purple
        "df": {"color": (35, 55, 75), "regular": True, "noise": 10}            # Dermatofibroma: firm/brown
    }
    
    generated_count = 0
    for idx, row in df.head(limit).iterrows():
        image_id = row['IMAGE ID']
        dx = row['DX'].lower()
        props = classes_props.get(dx, {"color": (40, 60, 90), "regular": True, "noise": 15})
        
        # 1. Base skin tone background
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        base_color = [np.random.randint(180, 210), np.random.randint(190, 220), np.random.randint(220, 250)]
        img[:, :] = base_color
        
        skin_noise = np.random.normal(0, 3, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + skin_noise, 0, 255).astype(np.uint8)
        
        # 2. Draw lesion
        center = (int(128 + np.random.randint(-15, 15)), int(128 + np.random.randint(-15, 15)))
        axes = (int(np.random.randint(30, 60)), int(np.random.randint(30, 60)))
        
        if not props["regular"]:
            num_sub_lesions = np.random.randint(3, 6)
            for _ in range(num_sub_lesions):
                sub_center = (int(center[0] + np.random.randint(-12, 12)), int(center[1] + np.random.randint(-12, 12)))
                sub_axes = (int(axes[0] + np.random.randint(-15, 5)), int(axes[1] + np.random.randint(-15, 5)))
                angle_val = int(np.random.randint(0, 180))
                
                color_var = (
                    int(np.clip(props["color"][0] + np.random.randint(-15, 15), 0, 255)),
                    int(np.clip(props["color"][1] + np.random.randint(-15, 15), 0, 255)),
                    int(np.clip(props["color"][2] + np.random.randint(-15, 15), 0, 255))
                )
                cv2.ellipse(img, sub_center, sub_axes, angle_val, 0, 360, color_var, -1)
        else:
            angle = int(np.random.randint(0, 180))
            color = props["color"]
            cv2.ellipse(img, center, axes, angle, 0, 360, color, -1)
            
        img = cv2.GaussianBlur(img, (9, 9), 0)
        
        # Save matching image_id
        filepath = os.path.join(image_dir, f"{image_id}.jpg")
        cv2.imwrite(filepath, img)
        generated_count += 1
        
    print(f"[+] Successfully generated {generated_count} synthetic lesion images.")

def load_data(dataset_excel, image_dir, limit=200):
    """
    Loads HAM10000 dataset, extracts features for Random Forest, and processes images for CNN.
    """
    if not os.path.exists(dataset_excel):
        print(f"[Error] Dataset metadata Excel file not found: {dataset_excel}")
        sys.exit(1)
        
    df = pd.read_excel(dataset_excel)
    print(f"\n--- Metadata Loaded: {len(df)} rows ---")
    
    # Class configuration
    class_mapping = dict(zip(df['DX'].str.lower(), df['DX LABEL']))
    class_names = sorted(list(set(df['DX LABEL'])))
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Limit rows to process
    df_subset = df.head(limit).copy()
    
    # Check if images exist, generate if not
    sample_img_id = df_subset.iloc[0]['IMAGE ID']
    sample_path = os.path.join(image_dir, f"{sample_img_id}.jpg")
    if not os.path.exists(sample_path):
        generate_mock_images_for_metadata(df, image_dir, limit)
        
    X_features = []
    X_pixels = []  # For CNN
    y = []
    
    print("\nProcessing images and extracting features...")
    processed_count = 0
    for idx, row in df_subset.iterrows():
        image_id = row['IMAGE ID']
        dx_label = row['DX LABEL']
        img_path = os.path.join(image_dir, f"{image_id}.jpg")
        
        if not os.path.exists(img_path):
            continue
            
        try:
            # 1. Feature Extraction (RF)
            features, _ = extract_features(img_path)
            X_features.append(features)
            
            # 2. Pixel Loading & Resizing (CNN)
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, (64, 64)) # Resize down for quick training
            img_normalized = img_resized / 255.0
            X_pixels.append(img_normalized)
            
            y.append(label_to_idx[dx_label])
            processed_count += 1
        except Exception as e:
            print(f"  [Warning] Failed to process image {image_id}: {e}")
            
    print(f"[+] Loaded and processed {processed_count} samples.")
    return np.array(X_features), np.array(X_pixels), np.array(y), class_names, label_to_idx, df

def train_rf(X_train, X_test, y_train, y_test, class_names, model_dir):
    """
    Trains and evaluates the Random Forest Classifier.
    """
    print("\n--- Training Random Forest Classifier ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, class_weight='balanced')
    clf.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Random Forest Accuracy : {acc*100:.2f}%")
    print(f"Random Forest F1-Score : {f1*100:.2f}%")
    
    # Save RF Model & Scaler using joblib
    joblib.dump(clf, os.path.join(model_dir, "model.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    
    return clf, y_pred, y_prob, acc, precision, recall, f1

def train_cnn(X_train_pix, X_test_pix, y_train, y_test, num_classes, model_dir):
    """
    Optional CNN Classifier implementation using TensorFlow.
    """
    print("\n--- Training Optional CNN Classifier ---")
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        
        # Build CNN
        model = Sequential([
            Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Train
        epochs = 10
        model.fit(X_train_pix, y_train, validation_data=(X_test_pix, y_test), epochs=epochs, batch_size=16, verbose=1)
        
        # Evaluate
        y_prob = model.predict(X_test_pix)
        y_pred = np.argmax(y_prob, axis=1)
        
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"CNN Accuracy : {acc*100:.2f}%")
        print(f"CNN F1-Score : {f1*100:.2f}%")
        
        # Save CNN model
        model.save(os.path.join(model_dir, "cnn_model.h5"))
        
        return model, y_pred, y_prob, acc, precision, recall, f1
    except ImportError:
        print("[!] TensorFlow not found on this system. Skipping CNN model training.")
        return None, None, None, 0.0, 0.0, 0.0, 0.0

def save_visualizations(y_test, rf_pred, cnn_pred, class_names, rf_clf, model_dir):
    """
    Generates and saves Seaborn/Matplotlib visual dashboards for the Streamlit UI.
    """
    os.makedirs(model_dir, exist_ok=True)
    sns.set_theme(style="darkgrid")
    
    # 1. Confusion Matrix Plot
    fig, axes = plt.subplots(1, 2 if cnn_pred is not None else 1, figsize=(14, 6))
    
    # RF Confusion Matrix
    rf_cm = confusion_matrix(y_test, rf_pred)
    ax_rf = axes[0] if cnn_pred is not None else axes
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax_rf)
    ax_rf.set_title("Random Forest Confusion Matrix")
    ax_rf.set_xlabel("Predicted")
    ax_rf.set_ylabel("True")
    
    # CNN Confusion Matrix (if trained)
    if cnn_pred is not None:
        cnn_cm = confusion_matrix(y_test, cnn_pred)
        sns.heatmap(cnn_cm, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names, ax=axes[1])
        axes[1].set_title("CNN Confusion Matrix")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")
        
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    
    # 2. RF Feature Importance Plot (first 15 features)
    if hasattr(rf_clf, 'feature_importances_'):
        plt.figure(figsize=(10, 5))
        # Total feature count is 32 (18 color, 8 texture, 6 shape)
        feature_labels = (
            [f"color_rgb_{c}_{m}" for c in ['R','G','B'] for m in ['mean','std','skew']] +
            [f"color_hsv_{c}_{m}" for c in ['H','S','V'] for m in ['mean','std','skew']] +
            [f"texture_glcm_d{d}_{p}" for d in [1,3] for p in ['contrast','correlation','energy','homogeneity']] +
            ['shape_area', 'shape_perimeter', 'shape_circularity', 'shape_solidity', 'shape_eccentricity']
        )
        # Handle matching size
        importances = rf_clf.feature_importances_
        if len(importances) == len(feature_labels):
            indices = np.argsort(importances)[::-1][:15]
            sns.barplot(x=importances[indices], y=[feature_labels[i] for i in indices], palette="viridis")
            plt.title("Top 15 Random Forest Feature Importances")
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, "feature_importance.png"), dpi=150)
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train Skin Cancer Detection Models on HAM10000 Dataset")
    parser.add_argument("--excel", type=str, default="HAM10000_dataset.xlsx", help="Metadata Excel sheet path")
    parser.add_argument("--image-dir", type=str, default="dataset/images", help="Lesion images directory")
    parser.add_argument("--model-dir", type=str, default="model_assets", help="Output model directory")
    parser.add_argument("--limit", type=int, default=250, help="Limit sample count for training")
    args = parser.parse_args()
    
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Load and process data
    X_features, X_pixels, y, class_names, label_to_idx, df_all = load_data(
        args.excel, args.image_dir, limit=args.limit
    )
    
    # Train-test split
    X_train_f, X_test_f, X_train_p, X_test_p, y_train, y_test = train_test_split(
        X_features, X_pixels, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 1. Train Random Forest
    rf_clf, rf_pred, rf_prob, rf_acc, rf_prec, rf_rec, rf_f1 = train_rf(
        X_train_f, X_test_f, y_train, y_test, class_names, args.model_dir
    )
    
    # 2. Train CNN (optional)
    cnn_model, cnn_pred, cnn_prob, cnn_acc, cnn_prec, cnn_rec, cnn_f1 = train_cnn(
        X_train_p, X_test_p, y_train, y_test, len(class_names), args.model_dir
    )
    
    # Save evaluation graphs
    save_visualizations(y_test, rf_pred, cnn_pred, class_names, rf_clf, args.model_dir)
    
    # Class Cancer mapping
    # Determine which classes are cancer (melanoma, bcc, akiec)
    cancer_classes = ["melanoma", "basal cell carcinoma", "actinic keratoses"]
    cancer_mapping = {}
    for name in class_names:
        is_cancer = any(kw in name.lower() for kw in cancer_classes) or "melanoma" in name.lower() or "carcinoma" in name.lower()
        cancer_mapping[name] = bool(is_cancer)
        
    # Save Metadata JSON
    metadata = {
        "classes": class_names,
        "label_to_idx": label_to_idx,
        "cancer_mapping": cancer_mapping,
        "rf_accuracy": float(rf_acc),
        "rf_precision": float(rf_prec),
        "rf_recall": float(rf_rec),
        "rf_f1": float(rf_f1),
        "cnn_trained": cnn_model is not None,
        "cnn_accuracy": float(cnn_acc) if cnn_model is not None else 0.0,
        "cnn_precision": float(cnn_prec) if cnn_model is not None else 0.0,
        "cnn_recall": float(cnn_rec) if cnn_model is not None else 0.0,
        "cnn_f1": float(cnn_f1) if cnn_model is not None else 0.0,
    }
    
    with open(os.path.join(args.model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("\n[+] All model training assets successfully generated and saved to model_assets/ folder.")

if __name__ == "__main__":
    main()
