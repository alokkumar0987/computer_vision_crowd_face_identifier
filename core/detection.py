
import os
import cv2
import numpy as np
import torch
from loguru import logger

# Import specific detectors from modules
from modules.detection.scrfd.detector import SCRFD

class FaceDetector:
    """
    Core Face Detection Service.
    
    This class provides a unified interface for various face detection models.
    Currently optimized for SCRFD (Sample and Computation Redistribution for Face Detection).
    """
    
    def __init__(self, model_path: str, threshold: float = 0.35, input_size: tuple = (640, 640)):
        """
        Initializes the face detector with the specified model and configurations.

        Args:
            model_path (str): Path to the ONNX model files.
            threshold (float): Confidence threshold for filtering detections.
            input_size (tuple): Target input resolution (width, height) for inference.
        """
        if not os.path.exists(model_path):
            error_msg = f"Detection model weights not found at: {model_path}"
            logger.critical(error_msg)
            raise FileNotFoundError(error_msg)
            
        # Initialize SCRFD detector
        # Future-proofing: This section can be extended to support YOLOv5-Face or RetinaFace
        self.detector = SCRFD(model_file=model_path)
        
        # Prepare the detector (defaulting to GPU if available)
        ctx_id = 0 if torch.cuda.is_available() else -1
        self.detector.prepare(ctx_id=ctx_id, input_size=input_size)
        
        self.threshold = threshold
        self.input_size = input_size
        
        logger.info(f"FaceDetector initialized successfully | Mode: {'GPU' if ctx_id >=0 else 'CPU'} | Thresh: {threshold}")

    def detect(self, frame: np.ndarray):
        """
        Performs face detection on the input frame.

        Args:
            frame (np.ndarray): The input video frame (BGR format).

        Returns:
            tuple: A pair containing:
                - bboxes (np.ndarray): Detected bounding boxes [x1, y1, x2, y2, score].
                - landmarks (np.ndarray): Detected facial landmarks [N, 5, 2].
        """
        if frame is None:
            return np.empty((0, 5)), []

        # We use detect_tracking to get raw floating point scores for better tracking stability
        det, _, _, landmarks = self.detector.detect_tracking(
            frame, 
            thresh=self.threshold, 
            input_size=self.input_size
        )
        
        # Ensure output is in expected numpy format
        bboxes = det.numpy() if isinstance(det, torch.Tensor) else det
        
        return bboxes, landmarks

    def set_threshold(self, threshold: float):
        """Updates the detection threshold at runtime."""
        self.threshold = threshold
        logger.info(f"FaceDetector threshold updated to: {threshold}")
