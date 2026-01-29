import time
import torch
import numpy as np
import cv2
from loguru import logger

# Import specific recognition models and alignment tools
from modules.recognition.arcface.model import iresnet_inference
from face_alignment.alignment import norm_crop

class RecognitionService:
    """
    Core Face Recognition Service.
    
    This class handles facial feature extraction (embeddings) and manages 
    per-track inference caching to optimize performance.
    
    Supported Models: iResNet (ArcFace)
    """
    
    def __init__(self, model_path: str, model_name: str = "r100", device: str = None):
        """
        Initializes the recognition service.

        Args:
            model_path (str): Path to the model weight files (.pth).
            model_name (str): Backbone name (e.g., 'r18', 'r34', 'r50', 'r100').
            device (str, optional): Target device ('cuda' or 'cpu'). Auto-detects if None.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Device fallback logic
        if self.device == "cuda" and not torch.cuda.is_available():
             logger.warning("CUDA requested but not available. Falling back to CPU for Recognition.")
             self.device = "cpu"

        # Initialize the model using the factory function
        try:
            self.model = iresnet_inference(model_name, model_path, device=self.device)
            logger.info(f"RecognitionService initialized | Model: {model_name} | Device: {self.device}")
        except Exception as e:
            logger.critical(f"Failed to initialize Recognition Model: {e}")
            raise

        self.cache = {}  # track_id -> last_inference_timestamp
        self.verify_interval = 1.0  # Minimum seconds between re-recognition for the same track

    def get_embedding(self, frame: np.ndarray, landmarks: np.ndarray, track_id: int) -> np.ndarray:
        """
        Extracts a facial embedding for the given track.
        
        Args:
            frame (np.ndarray): The source video frame.
            landmarks (np.ndarray): 5 facial landmarks for alignment.
            track_id (int): Unique identifier for the track.
        
        Returns:
            np.ndarray or None: A 512-dimensional normalized embedding or None if cached/failed.
        """
        current_time = time.time()
        
        # Performance Optimization: Check if we recently processed this track
        if track_id in self.cache:
            if current_time - self.cache[track_id] < self.verify_interval:
                return None
        
        try:
            # Step 1: Face Alignment (Standard 112x112 crop)
            aligned = norm_crop(frame, landmarks)
            
            # Step 2: Pre-processing for ArcFace (iresnet)
            # Standard InsightFace normalization: (Input - 127.5) / 128.0
            input_blob = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            input_blob = np.transpose(input_blob, (2, 0, 1)) # HWC -> CHW
            input_blob = torch.from_numpy(input_blob).unsqueeze(0).float()
            
            # Map [0, 255] to roughly [-1, 1]
            input_blob.div_(255).sub_(0.5).div_(0.5) 
            
            input_blob = input_blob.to(self.device)
            
            # Step 3: Forward Pass
            with torch.no_grad():
                embedding = self.model(input_blob)
                embedding = embedding.cpu().numpy()[0]
            
            # Update cache and return the result
            self.cache[track_id] = current_time
            return embedding
            
        except Exception as e:
            logger.error(f"Recognition failed for track_id {track_id}: {e}")
            return None

    def clear_cache(self, track_id: int = None):
        """Clears inference cache for a specific track or all tracks."""
        if track_id is not None:
            self.cache.pop(track_id, None)
        else:
            self.cache.clear()
