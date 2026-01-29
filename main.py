"""
Face Recognition Pipeline - Production Entry Point

PRIVATE & CONFIDENTIAL
This software is the proprietary property of Alok Kumar.

Author: Alok Kumar
Date: January 2026
"""

import os
import cv2
import time
import yaml
import torch
import numpy as np
from loguru import logger

# Prevent OpenMP library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Core Modules
from core.ingestion import CameraIngestion
from core.detection import FaceDetector
from core.tracking import FaceTracker
from core.quality import QualityGate
from core.recognition import RecognitionService
from core.search import VectorSearch
from core.identity import IdentityManager
from core.output import OutputManager

# --- PIPELINE CONFIGURATION ---
SETTINGS = {
    "DETECTION_INTERVAL": 1,        # 1 = Every frame (maximum tracking sensitivity)
    "MOTION_THRESHOLD": 500.0,      # Pixels/sec (Skip recognition if moving too fast)
    "QUALITY_THRESHOLD": 40.0,      # Min score for 'Best Frame' selection
    "RECOGNITION_CONFIDENCE": 0.25, # FAISS distance threshold
    "VOTING_WINDOW": 5              # Frames to stabilize identity
}

def match_metrics(tracks, detections, landmarks):
    """
    Associates track IDs with detected facial landmarks using Intersection over Union (IoU).
    """
    track_to_landmarks = {}
    if len(tracks) == 0 or len(detections) == 0:
        return track_to_landmarks
        
    det_bboxes = detections[:, :4]
    
    for t in tracks:
        tx1, ty1, tx2, ty2 = t[:4]
        tid = int(t[4])
        
        # Calculate IoU with all detections
        # intersection
        ix1 = np.maximum(tx1, det_bboxes[:, 0])
        iy1 = np.maximum(ty1, det_bboxes[:, 1])
        ix2 = np.minimum(tx2, det_bboxes[:, 2])
        iy2 = np.minimum(ty2, det_bboxes[:, 3])
        
        iw = np.maximum(0, ix2 - ix1)
        ih = np.maximum(0, iy2 - iy1)
        intersect = iw * ih
        
        area_t = (tx2 - tx1) * (ty2 - ty1)
        area_d = (det_bboxes[:, 2] - det_bboxes[:, 0]) * (det_bboxes[:, 3] - det_bboxes[:, 1])
        
        union = area_t + area_d - intersect
        iou = intersect / (union + 1e-6)
        
        best_idx = np.argmax(iou)
        if iou[best_idx] > 0.5:
            track_to_landmarks[tid] = landmarks[best_idx]
            
    return track_to_landmarks

def main():
    """Main execution loop for the Face Recognition Pipeline."""
    
    # 1. Load System Configuration
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Config load failed: {e}. Using defaults.")
        config = {
            "camera": {"source": 0, "fps": 30},
            "models": {
                "detection": "modules/detection/scrfd/weights/scrfd_2.5g_bnkps.onnx",
                "recognition": "modules/recognition/arcface/weights/arcface_r100.pth"
            },
            "watchlist": []
        }
    
    # 2. Initialize Hardware & Model Layers
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ingestion = CameraIngestion(source=config["camera"]["source"], target_fps=config["camera"]["fps"])
    detector = FaceDetector(config["models"]["detection"])
    tracker = FaceTracker(frame_rate=config["camera"]["fps"])
    quality = QualityGate()
    recognition = RecognitionService(config["models"]["recognition"], device=device)
    search = VectorSearch()
    identity_mgr = IdentityManager(voting_window=SETTINGS["VOTING_WINDOW"])
    output_mgr = OutputManager(watchlist=config["watchlist"])
    
    # Start Video Stream
    ingestion.start()
    logger.info("Pipeline Execution Started. Terminate with 'q'.")
    
    frame_id = 0
    try:
        while True:
            # Step A: Image Ingestion
            ret, frame, ts = ingestion.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            frame_id += 1
            h, w = frame.shape[:2]
            
            # Step B: Detection (Periodic or Continuous)
            if frame_id % SETTINGS["DETECTION_INTERVAL"] == 0 or frame_id == 1:
                bboxes, landmarks = detector.detect(frame)
            else:
                bboxes, landmarks = np.empty((0, 5)), []
            
            # Step C: Tracking Persistence
            tracks = tracker.update(bboxes, img_info=(h, w), img_size=(640, 640))
            track_landmarks_map = match_metrics(tracks, bboxes, landmarks)
            
            current_identities = {}
            
            # Step D: Behavioral Processing per Track
            for t in tracks:
                tid = int(t[4])
                x1, y1, x2, y2 = map(int, t[:4])
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                # Check if Identity is already locked for this track
                if identity_mgr.is_locked(tid):
                    confirmed_id, confirmed_score = identity_mgr.get_identity(tid)
                    current_identities[tid] = (confirmed_id, confirmed_score)
                    continue

                # Face Crop & Bounds Safety
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1: continue
                face_crop = frame[y1:y2, x1:x2]
                
                # Evaluation: Motion & Quality
                speed = tracker.compute_motion_speed(tid, (cx, cy))
                lms = track_landmarks_map.get(tid)
                passed_gate, quality_score = quality.check(face_crop, lms)
                
                # Temporal Memory: Update best frame if current is sharper
                if passed_gate:
                    identity_mgr.update_best_frame(tid, face_crop, quality_score, lms)

                # Step E: Event-Based Recognition Trigger
                can_run_recog = (
                    speed < SETTINGS["MOTION_THRESHOLD"] and 
                    quality_score > SETTINGS["QUALITY_THRESHOLD"] and 
                    lms is not None
                )
                
                if can_run_recog:
                    best_face, best_lms = identity_mgr.get_best_frame(tid)
                    if best_face is not None:
                        # Extract 512-d Deep Feature
                        embedding = recognition.get_embedding(frame, best_lms, tid)
                        if embedding is not None:
                            # FAISS Search
                            name, score = search.search(embedding, threshold=SETTINGS["RECOGNITION_CONFIDENCE"])
                            confirmed_id, confirmed_score = identity_mgr.update(tid, name, score)
                        else:
                            confirmed_id, confirmed_score = identity_mgr.get_identity(tid)
                    else:
                        confirmed_id, confirmed_score = identity_mgr.get_identity(tid)
                else:
                    confirmed_id, confirmed_score = identity_mgr.get_identity(tid)
                    
                current_identities[tid] = (confirmed_id, confirmed_score)
            
            # Step F: Visualization & Alerts
            vis_frame = output_mgr.draw(frame, tracks, current_identities)
            cv2.imshow("Face Recognition Pipeline", vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logger.info("Pipeline Interrupted by User.")
    finally:
        ingestion.stop()
        cv2.destroyAllWindows()
        logger.info("Pipeline Shutdown Gracefully.")

if __name__ == "__main__":
    main()
