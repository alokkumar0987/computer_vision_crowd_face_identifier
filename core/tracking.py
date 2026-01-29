
import numpy as np
from loguru import logger
from modules.tracking.tracker.byte_tracker import BYTETracker

class FaceTracker:
    """
    Wraps ByteTrack for face tracking.
    """
    def __init__(self, frame_rate=10, track_buffer=180, track_thresh=0.5, match_thresh=0.8):
        """
        Args:
            frame_rate (int): Processing FPS.
            track_buffer (int): Buffer size in terms of 30FPS frames.
                               To get actual buffer size of 60 frames at 10FPS:
                               buffer = (10/30) * 180 = 60.
            track_thresh (float): Threshold for high quality detection.
            match_thresh (float): IOU matching threshold.
        """
        args = {
            "track_thresh": track_thresh,
            "track_buffer": track_buffer,
            "match_thresh": match_thresh,
            "mot20": False
        }
        self.tracker = BYTETracker(args, frame_rate=frame_rate)
        self.track_history = {} # tid -> [(center, timestamp), ...]
        logger.info(f"FaceTracker initialized. Buffer: {self.tracker.buffer_size} frames.")

    def compute_motion_speed(self, track_id: int, center: tuple) -> float:
        """
        Calculates motion speed in pixels per second.
        """
        import time
        now = time.time()
        
        if track_id not in self.track_history:
            self.track_history[track_id] = [(center, now)]
            return 0.0
            
        prev_center, prev_time = self.track_history[track_id][-1]
        dt = max(now - prev_time, 1e-6)
        
        # Euclidean distance
        dist = np.linalg.norm(np.array(center) - np.array(prev_center))
        speed = dist / dt
        
        # Update history
        self.track_history[track_id].append((center, now))
        # Keep only last 5 entries
        if len(self.track_history[track_id]) > 5:
            self.track_history[track_id].pop(0)
            
        return speed

    def remove_history(self, track_id: int):
        if track_id in self.track_history:
            del self.track_history[track_id]

    def update(self, bboxes: np.ndarray, img_info: tuple, img_size: tuple):
        """
        Update tracks with new detections.
        
        Args:
            bboxes (np.ndarray): [x1, y1, x2, y2, score]
            img_info (tuple): (height, width)
            img_size (tuple): (height, width) - input size to model ? 
                              Actually BYTETracker update uses these to scale back? 
                              Let's check detector output.
        Returns:
            list: List of tracks [x1, y1, x2, y2, track_id, score]
        """
        if bboxes is None or len(bboxes) == 0:
            # Handle empty detection
            # Create a dummy empty array of shape (0, 5) to pass to update?
            # BYTETracker expects (N, 5)
             bboxes = np.empty((0, 5))

        # BYTETracker update() signature: update(output_results, img_info, img_size)
        # output_results: tensor or numpy [N, 5]
        # img_info: [h, w, ...]
        # img_size: [h, w] (model input size) used for scaling if detections are normalized?
        # In our detector, we return bboxes already in image coordinates?
        # SCRFD `detect` returns bboxes in original image scale if we look at `detect` code (it divides by det_scale).
        # So we don't need further scaling?
        # BYTETracker update line 184: scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        # It calculates scale and divides bboxes by it.
        # This assumes bboxes are in "img_size" scale (model scale) and need to be valid for img_info (original).
        # BUT wait: `bboxes /= scale`.
        # If `img_size` == `img_info` (if we pass same), scale is 1.
        # Our detector returns bboxes in original image coordinates.
        # So we should pass img_size = img_info to avoid scaling inside tracker, OR pre-scale them?
        # Let's pass img_size = img_info so scale is 1.0.
        
        tracks = self.tracker.update(bboxes, img_info, img_info)
        
        results = []
        for t in tracks:
            tlwh = t.tlwh
            tid = t.track_id
            score = t.score
            # Convert tlwh to tlbr
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            results.append([x1, y1, x2, y2, tid, score])
            
        return results
