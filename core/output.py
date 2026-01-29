import cv2
import numpy as np
import time
import os
import winsound
import pyttsx3
import threading
from loguru import logger

class OutputManager:
    """
    Manages visualization and alerting for the face recognition pipeline.
    
    Responsibilities:
    - Overlaying bounding boxes and identity labels on video frames.
    - Managing a watchlist and triggering audio/visual alerts.
    - Handling Text-to-Speech (TTS) notifications.
    """
    
    def __init__(self, watchlist=None, alert_sound="alert.wav", alert_cooldown=5.0):
        """
        Initializes the OutputManager.

        Args:
            watchlist (list, optional): List of names to trigger alerts for.
            alert_sound (str): Path to the WAV file for audio alerts.
            alert_cooldown (float): Minimum seconds between consecutive alerts.
        """
        self.watchlist = [name.lower() for name in (watchlist or [])]
        self.alert_sound = alert_sound
        self.alert_cooldown = alert_cooldown
        self.last_alert_time = 0
        
        # Initialize TTS Engine settings
        self.tts_rate = 150
        self.tts_volume = 1.0
        
        logger.info(f"OutputManager initialized | Watchlist Size: {len(self.watchlist)}")

    def draw(self, frame: np.ndarray, tracks: list, identities: dict) -> np.ndarray:
        """
        Draws tracking boxes and identity markers on the frame.

        Args:
            frame (np.ndarray): The raw video frame.
            tracks (list): List of active tracks from the tracker.
            identities (dict): Mapping of track_id to (identity_name, confidence_score).

        Returns:
            np.ndarray: The annotated frame for display.
        """
        vis_frame = frame.copy()
        
        for t in tracks:
            # t scale: [x1, y1, x2, y2, track_id, score]
            x1, y1, x2, y2 = map(int, t[:4])
            tid = int(t[4])
            
            # Fetch Identity Information
            identity_data = identities.get(tid, ("Unknown", 0.0))
            identity, score = identity_data if isinstance(identity_data, tuple) else (identity_data, 0.0)
            
            # Determine Color Scheme
            is_watchlist = identity.lower() in self.watchlist
            if is_watchlist:
                color = (0, 0, 255)  # Red for Alerts
            elif identity == "Unknown":
                color = (0, 255, 255) # Yellow for Unknowns
            elif identity == "LowQuality":
                color = (128, 128, 128) # Grey for low-quality frames
            else:
                color = (0, 255, 0)  # Green for Recognized
            
            # Draw Bounding Box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Construct Label
            if identity == "LowQuality":
                label = f"ID:{tid} Processing..."
            elif identity == "Unknown":
                label = f"ID:{tid} {identity}"
            else:
                label = f"ID:{tid} {identity} ({score:.2f})"
            
            # Draw Text Background
            (l_w, l_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(vis_frame, (x1, y1 - 25), (x1 + l_w, y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Trigger Alert for Watchlist Hits
            if is_watchlist:
                self.trigger_alert(identity, tid)
                
        return vis_frame

    def trigger_alert(self, identity: str, track_id: int):
        """Triggers audio and voice notifications for watchlist targets."""
        current_time = time.time()
        if current_time - self.last_alert_time > self.alert_cooldown:
            logger.warning(f"WATCHLIST DETECTED: {identity} (Track {track_id})")
            
            # 1. Play Alarm Sound (Non-blocking)
            if os.path.exists(self.alert_sound):
                try:
                    winsound.PlaySound(self.alert_sound, winsound.SND_FILENAME | winsound.SND_ASYNC)
                except Exception as e:
                    logger.debug(f"Audio playback error: {e}")
            
            # 2. Start TTS Notification in a daemon thread
            alert_text = f"Alert: {identity} is detected."
            threading.Thread(target=self._speak, args=(alert_text,), daemon=True).start()
            
            self.last_alert_time = current_time

    def _speak(self, text: str):
        """Internal worker function for Text-to-Speech."""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', self.tts_rate)
            engine.setProperty('volume', self.tts_volume)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            logger.debug(f"TTS Engine error: {e}")

    def log_event(self, message: str):
        """Logs a custom event to the system log."""
        logger.info(f"Pipeline Event: {message}")
