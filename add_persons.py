"""
Face Enrollment Tool - Adding Persons to Database

Author: Alok Kumar
Date: 2026-01-29
"""
import argparse
import os
import shutil

import cv2
import numpy as np
from loguru import logger

# Constants for easy model switching in the future (RetinaFace, YOLOv5-Face, SCRFD)
DETECTOR_PATH = "modules/detection/scrfd/weights/scrfd_2.5g_bnkps.onnx"
RECOGNITION_PATH = "modules/recognition/arcface/weights/arcface_r100.pth"

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
logger.info(f"Initializing models on {device}...")
detector = SCRFD(model_file=DETECTOR_PATH)
recognizer = iresnet_inference(model_name="r100", path=RECOGNITION_PATH, device=str(device))

@torch.no_grad()
def get_feature(face_image: np.ndarray) -> np.ndarray:
    """Extracts and normalizes facial features."""
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

def add_persons(backup_dir: str, add_persons_dir: str, faces_save_dir: str, features_path: str):
    """
    Main entry for adding new persons. 
    Reads raw images, detects faces, crops them, and generates embeddings.
    """
    images_name = []
    images_emb = []

    if not os.path.exists(add_persons_dir):
        logger.error(f"Directory not found: {add_persons_dir}")
        return

    for name_person in os.listdir(add_persons_dir):
        person_image_path = os.path.join(add_persons_dir, name_person)
        if not os.path.isdir(person_image_path): continue

        logger.info(f"Processing new person: {name_person}")
        person_face_path = os.path.join(faces_save_dir, name_person)
        os.makedirs(person_face_path, exist_ok=True)

        for image_name in os.listdir(person_image_path):
            if not image_name.lower().endswith(("png", "jpg", "jpeg")): continue
            
            img_path = os.path.join(person_image_path, image_name)
            input_image = cv2.imread(img_path)
            if input_image is None: continue

            # Face Detection
            bboxes, _ = detector.detect(image=input_image)

            for i in range(len(bboxes)):
                h, w = input_image.shape[:2]
                x1, y1, x2, y2, score = bboxes[i]
                
                # Bounds Safety
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))

                if x2 <= x1 or y2 <= y1: continue

                # Crop & Save
                face_image = input_image[y1:y2, x1:x2]
                if face_image.size == 0: continue

                number_files = len(os.listdir(person_face_path))
                path_save_face = os.path.join(person_face_path, f"{number_files}.jpg")
                cv2.imwrite(path_save_face, face_image)

                # Feature Extraction
                embedding = get_feature(face_image)
                images_emb.append(embedding)
                images_name.append(name_person)

    if not images_name:
        logger.warning("No new faces detected in the provided directory.")
        return

    # Database Update
    images_emb = np.array(images_emb)
    images_name = np.array(images_name)

    features = read_features(features_path)
    if features is not None:
        old_images_name, old_images_emb = features
        images_name = np.hstack((old_images_name, images_name))
        images_emb = np.vstack((old_images_emb, images_emb))
        logger.info("Updating existing feature database...")

    np.savez_compressed(features_path, images_name=images_name, images_emb=images_emb)
    logger.info(f"Features saved successfully to: {features_path}.npz")

    # Cleanup: Move processed images to backup
    for sub_dir in os.listdir(add_persons_dir):
        dir_to_move = os.path.join(add_persons_dir, sub_dir)
        dest_dir = os.path.join(backup_dir, sub_dir)
        
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir) if os.path.isdir(dest_dir) else os.remove(dest_dir)
        
        shutil.move(dir_to_move, backup_dir)
        logger.debug(f"Moved {sub_dir} to backup.")

    logger.success(f"Person addition complete.")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backup-dir",
        type=str,
        default="./datasets/backup",
        help="Directory to save person data.",
    )
    parser.add_argument(
        "--add-persons-dir",
        type=str,
        default="./datasets/new_persons",
        help="Directory to add new persons.",
    )
    parser.add_argument(
        "--faces-save-dir",
        type=str,
        default="./datasets/data/",
        help="Directory to save faces.",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default="./datasets/face_features/feature",
        help="Path to save face features.",
    )
    opt = parser.parse_args()

    # Run the main function
    add_persons(**vars(opt))
