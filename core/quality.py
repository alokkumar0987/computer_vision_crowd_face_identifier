
import cv2
import numpy as np
from loguru import logger

class QualityGate:
    """
    Filters faces based on quality metrics:
    - Minimum Head Size
    - Blur (Laplacian Variance)
    - Pose (Heuristic based on landmarks)
    """
    def __init__(self, min_size=30, min_blur_score=20.0):
        self.min_size = min_size
        self.min_blur_score = min_blur_score

    def check(self, face_img: np.ndarray, landmarks: np.ndarray = None) -> (bool, float):
        """
        Returns (passed, score). 
        score is a weighted fusion of quality metrics [0-100].
        """
        # 1. Size Check
        h, w = face_img.shape[:2]
        if h < self.min_size or w < self.min_size:
            return False, 0.0

        # 2. Blur Calculation
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize blur score (heuristic: 100.0 is very sharp)
        norm_blur = min(1.0, blur_score / 100.0)

        # 3. Pose Calculation
        norm_pose = 1.0 # Default perfect pose
        if landmarks is not None and len(landmarks) == 5:
            leye = landmarks[0]
            reye = landmarks[1]
            nose = landmarks[2]
            
            d_l = np.linalg.norm(leye - nose)
            d_r = np.linalg.norm(reye - nose)
            ratio = max(d_l, d_r) / (min(d_l, d_r) + 1e-6)
            
            # Normalize pose ratio (heuristic: 1.0 is front, 3.5+ is side)
            # 1.0 -> 1.0, 3.5 -> 0.0
            norm_pose = max(0.0, (3.5 - ratio) / 2.5)
        
        # Weighted Fusion (0.4 Blur + 0.6 Pose)
        final_score = (0.4 * norm_blur + 0.6 * norm_pose) * 100.0
        
        # Threshold Check
        passed = (blur_score >= self.min_blur_score) and (norm_pose > 0.0)
        
        return passed, final_score
