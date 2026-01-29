
import cv2
import time
import threading
import queue
from typing import Optional, Tuple
from loguru import logger

class CameraIngestion:
    """
    Handles camera ingestion from RTSP streams or local video files.
    Enforces a target FPS by skipping frames if necessary.
    """
    def __init__(self, source: str, target_fps: int = 10, queue_size: int = 30):
        """
        Args:
            source (str): RTSP URL or video file path.
            target_fps (int): Desired output frames per second (8-12 recommended).
            queue_size (int): Size of the frame buffer queue.
        """
        self.source = source
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = None
        self.cap = None
        
    def start(self):
        """Starts the ingestion thread."""
        logger.info(f"Starting camera ingestion for source: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error(f"Failed to open source: {self.source}")
            return
            
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        """Internal loop to read frames and push to queue at target FPS."""
        last_time = time.time()
        
        while not self.stopped and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"Failed to read frame from {self.source}, retrying...")
                # Simple retry logic or break depending on requirement. 
                # For RTSP, we might want to reconnect. For file, break.
                if isinstance(self.source, int) or str(self.source).isdigit(): # Webcam
                    break
                # Check if it's a file
                 # Simple heuristic: if it's a file path and we reached the end, loop or stop.
                 # For now, let's assume valid stream and if it breaks, we stop.
                break 

            current_time = time.time()
            elapsed = current_time - last_time
            
            # Simple Frame Limiting
            if elapsed >= self.frame_interval:
                if not self.queue.full():
                    # Add simple timestamp for sync
                    self.queue.put((frame, current_time))
                    last_time = current_time
                else:
                    # Drop frame if queue is full (backpressure)
                    pass
            else:
                # If we are reading too fast, we simply don't put it in queue (skip)
                # But notice cap.read() consumes the frame, so we are effectively skipping.
                # To reduce CPU usage on file, we might sleep, but for RTSP, we must keep reading to empty buffer.
                pass

        self.cap.release()

    def read(self) -> Tuple[bool, Optional[object], float]:
        """
        Returns:
            start_status (bool): True if frame is valid.
            frame (np.array): The image frame.
            timestamp (float): Unix timestamp of capture.
        """
        try:
            frame, timestamp = self.queue.get(timeout=2.0)
            return True, frame, timestamp
        except queue.Empty:
            return False, None, 0.0

    def stop(self):
        """Stops the ingestion thread."""
        self.stopped = True
        if self.thread:
            self.thread.join()
        logger.info("Camera ingestion stopped")
