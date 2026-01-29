"""
Face Feature Maintenance Tool - Regenerating Databases

Author: Alok Kumar
Date: 2026-01-29
"""
import argparse
import os

import cv2
import numpy as np
import torch
from torchvision import transforms

from modules.recognition.arcface.model import iresnet_inference

from loguru import logger

# Model weighting constants
RECOGNITION_PATH = "modules/recognition/arcface/weights/arcface_r100.pth"

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
logger.info(f"Initializing recognition model on {device} for feature regeneration...")
recognizer = iresnet_inference(model_name="r100", path=RECOGNITION_PATH, device=str(device))

@torch.no_grad()
def get_feature(face_image: np.ndarray) -> np.ndarray:
    """Extracts and normalizes facial features from existing face crops."""
    face_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)
    
    emb_img_face = recognizer(face_image)[0].cpu().numpy()
    images_emb = emb_img_face / (np.linalg.norm(emb_img_face) + 1e-6)
    return images_emb

def regenerate_features(faces_dir: str, features_path: str):
    """
    Scans the processed face directory and regenerates the feature database.
    Useful when manually modifying face crops in the data directory.
    """
    images_name = []
    images_emb = []

    if not os.path.exists(faces_dir):
        logger.error(f"Faces directory not found: {faces_dir}")
        return

    logger.info(f"Scanning face directory: {faces_dir}")
    
    # Iterate through each person's folder
    for name_person in os.listdir(faces_dir):
        person_face_path = os.path.join(faces_dir, name_person)
        if not os.path.isdir(person_face_path): continue
            
        logger.info(f"Regenerating features for: {name_person}")
        face_count = 0

        for image_name in os.listdir(person_face_path):
            if not image_name.lower().endswith(("png", "jpg", "jpeg")): continue
            
            img_path = os.path.join(person_face_path, image_name)
            face_image = cv2.imread(img_path)
                
            if face_image is None or face_image.size == 0:
                logger.warning(f"Skipping unreadable image: {image_name}")
                continue

            try:
                embedding = get_feature(face_image)
                images_emb.append(embedding)
                images_name.append(name_person)
                face_count += 1
            except Exception as e:
                logger.error(f"Error processing {image_name}: {e}")
                continue
        
        logger.debug(f"Added {face_count} embeddings for {name_person}")

    if not images_name:
        logger.warning("No face images found to process.")
        return

    # Save regenerated database
    images_emb = np.array(images_emb)
    images_name = np.array(images_name)
    np.savez_compressed(features_path, images_name=images_name, images_emb=images_emb)

    logger.success(f"Successfully regenerated features for {len(set(images_name))} person(s).")
    logger.info(f"Total embeddings in database: {len(images_name)}")
    logger.info(f"Database saved to: {features_path}.npz")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--faces-dir",
        type=str,
        default="./datasets/data/",
        help="Directory containing face images.",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default="./datasets/face_features/feature",
        help="Path to save face features.",
    )
    opt = parser.parse_args()

    # Run the main function
    regenerate_features(**vars(opt))
