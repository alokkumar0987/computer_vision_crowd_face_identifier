
from collections import deque
from collections import Counter
import numpy as np
from loguru import logger

class IdentityManager:
    """
    Manages Track-to-Identity binding.
    Uses voting to stabilize identity against flicker.
    """
    def __init__(self, voting_window=5, confirm_threshold=0.6):
        self.tracks = {} # track_id -> {history, confirmed_id, locked, best_face, best_quality}
        self.voting_window = voting_window
        self.confirm_threshold = confirm_threshold

    def update_best_frame(self, track_id: int, face_img: np.ndarray, quality_score: float, landmarks: np.ndarray):
        """
        Updates the best face image for this track if current quality is higher.
        """
        if track_id not in self.tracks:
            self.tracks[track_id] = {
                "history": deque(maxlen=self.voting_window),
                "confirmed_id": "Unknown",
                "locked": False,
                "best_face": face_img,
                "best_quality": quality_score,
                "best_lms": landmarks
            }
        else:
            state = self.tracks[track_id]
            if quality_score > state.get("best_quality", 0.0):
                state["best_quality"] = quality_score
                state["best_face"] = face_img
                state["best_lms"] = landmarks

    def is_locked(self, track_id: int) -> bool:
        return self.tracks.get(track_id, {}).get("locked", False)

    def get_best_frame(self, track_id: int):
        state = self.tracks.get(track_id, {})
        return state.get("best_face"), state.get("best_lms")

    def update(self, track_id: int, identity: str = None, score: float = 0.0):
        """
        Update track with new recognition result.
        If identity is None (skipped recognition), we trust existing state.
        """
        if track_id not in self.tracks:
            self.tracks[track_id] = {
                "history": deque(maxlen=self.voting_window),
                "confirmed_id": "Unknown",
                "locked": False,
                "last_score": 0.0
            }
        
        state = self.tracks[track_id]
        
        if identity is not None:
            # We have a fresh recognition
            state["history"].append((identity, score))
            state["last_score"] = score # Store last seen score
            self._recalculate(track_id)
        
        return state["confirmed_id"], state["last_score"]

    def _recalculate(self, track_id):
        state = self.tracks[track_id]
        history = state["history"]
        
        if not history:
            return

        # Simple Majority Voting (ignoring Unknowns if possible?)
        # Or Just strict majority
        
        counts = Counter([h[0] for h in history if h[0] != "Unknown"])
        
        if not counts:
             # All unknowns
             return
             
        top_id, count = counts.most_common(1)[0]
        
        # If we have enough votes
        if count >= (len(history) / 2):
            if state["confirmed_id"] != top_id:
                logger.info(f"Track {track_id} identity switch: {state['confirmed_id']} -> {top_id}")
            state["confirmed_id"] = top_id
            state["locked"] = True
        
        # ID Switch Recovery logic could be more complex (e.g. requiring consecutive mismatch to unlock)
        # Current logic allows switching if voting window shifts.

    def get_identity(self, track_id):
        state = self.tracks.get(track_id, {})
        return state.get("confirmed_id", "Unknown"), state.get("last_score", 0.0)

    def remove_track(self, track_id):
        if track_id in self.tracks:
            del self.tracks[track_id]
