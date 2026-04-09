"""Gesture detection and hand landmark processing for Air Drawing."""

from typing import Optional, Tuple, List
from collections import deque
import numpy as np

from config import (
    LANDMARK_INDEX_FINGER_TIP, LANDMARK_INDEX_FINGER_PIP,
    LANDMARK_MIDDLE_FINGER_TIP, LANDMARK_MIDDLE_FINGER_PIP,
    LANDMARK_RING_FINGER_TIP, LANDMARK_RING_FINGER_DIP,
    LANDMARK_PINKY_TIP, LANDMARK_PINKY_DIP,
    DEFAULT_SMOOTH_BUFFER_SIZE
)


class GestureDetector:
    """Detects hand gestures and determines drawing mode."""
    
    MODE_DRAW = "DRAW"
    MODE_STOP = "STOP"
    MODE_CLEAR = "CLEAR"
    
    def __init__(self):
        self.clear_start_time: Optional[float] = None
    
    @staticmethod
    def is_finger_up(
        hand_landmarks, 
        tip_id: int, 
        pip_id: int
    ) -> bool:
        """Returns True if the finger is extended (tip y < pip y)."""
        return hand_landmarks[tip_id].y < hand_landmarks[pip_id].y
    
    def detect_gesture(
        self, 
        hand_landmarks
    ) -> str:
        """
        Detect gesture from hand landmarks and return current mode.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            
        Returns:
            Current mode string: "DRAW", "STOP", or "CLEAR"
        """
        import time
        
        index_up = self.is_finger_up(
            hand_landmarks, 
            LANDMARK_INDEX_FINGER_TIP, 
            LANDMARK_INDEX_FINGER_PIP
        )
        middle_up = self.is_finger_up(
            hand_landmarks, 
            LANDMARK_MIDDLE_FINGER_TIP, 
            LANDMARK_MIDDLE_FINGER_PIP
        )
        ring_up = self.is_finger_up(
            hand_landmarks, 
            LANDMARK_RING_FINGER_TIP, 
            LANDMARK_RING_FINGER_DIP
        )
        pinky_up = self.is_finger_up(
            hand_landmarks, 
            LANDMARK_PINKY_TIP, 
            LANDMARK_PINKY_DIP
        )
        
        # Determine mode based on finger combination
        if index_up and not middle_up and not ring_up and not pinky_up:
            return self.MODE_DRAW
        elif index_up and middle_up and not ring_up and not pinky_up:
            return self.MODE_STOP
        elif index_up and middle_up and ring_up and pinky_up:
            return self._handle_clear_gesture()
        else:
            self.clear_start_time = None
            return self.MODE_STOP
    
    def _handle_clear_gesture(self) -> str:
        """Handle the clear canvas gesture with timing."""
        import time
        
        if self.clear_start_time is None:
            self.clear_start_time = time.time()
            return self.MODE_STOP
        
        from config import CLEAR_GESTURE_DURATION
        if time.time() - self.clear_start_time > CLEAR_GESTURE_DURATION:
            self.clear_start_time = None
            return self.MODE_CLEAR
        
        return self.MODE_STOP
    
    def reset(self):
        """Reset gesture detector state."""
        self.clear_start_time = None


class LandmarkProcessor:
    """Processes hand landmarks for coordinate extraction and smoothing."""
    
    def __init__(self, buffer_size: int = DEFAULT_SMOOTH_BUFFER_SIZE):
        self.buffer_size = buffer_size
        self.smooth_buffer: deque = deque(maxlen=buffer_size)
    
    def extract_fingertip(
        self, 
        hand_landmarks, 
        frame_width: int, 
        frame_height: int,
        landmark_index: int = LANDMARK_INDEX_FINGER_TIP
    ) -> Tuple[int, int]:
        """
        Extract fingertip coordinates in pixel space.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            frame_width: Width of the video frame
            frame_height: Height of the video frame
            landmark_index: Index of the landmark to extract
            
        Returns:
            Tuple of (x, y) pixel coordinates
        """
        landmark = hand_landmarks[landmark_index]
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        return (x, y)
    
    def smooth_coordinates(
        self, 
        x: int, 
        y: int
    ) -> Tuple[int, int]:
        """
        Apply smoothing to coordinates using a moving average buffer.
        
        Args:
            x: Raw x coordinate
            y: Raw y coordinate
            
        Returns:
            Tuple of smoothed (x, y) coordinates
        """
        self.smooth_buffer.append((x, y))
        sx = int(np.mean([p[0] for p in self.smooth_buffer]))
        sy = int(np.mean([p[1] for p in self.smooth_buffer]))
        return (sx, sy)
    
    def reset(self):
        """Reset the smoothing buffer."""
        self.smooth_buffer.clear()
