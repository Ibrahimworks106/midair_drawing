"""Camera management for Air Drawing application."""

from typing import Optional, Tuple
import cv2
import logging

from config import DEFAULT_CAMERA_INDEX, FALLBACK_CAMERA_INDEX

logger = logging.getLogger(__name__)


class CameraManager:
    """Manages webcam capture and frame processing."""
    
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_index: int = -1
    
    def open(self, preferred_index: int = DEFAULT_CAMERA_INDEX) -> bool:
        """
        Open the camera.
        
        Args:
            preferred_index: Preferred camera index to try first
            
        Returns:
            True if camera opened successfully, False otherwise
        """
        # Try preferred index first
        if self._open_camera(preferred_index):
            self.camera_index = preferred_index
            logger.info(f"Camera {preferred_index} opened successfully")
            return True
        
        # Fallback to alternative index
        logger.info(f"Trying fallback camera index {FALLBACK_CAMERA_INDEX}")
        if self._open_camera(FALLBACK_CAMERA_INDEX):
            self.camera_index = FALLBACK_CAMERA_INDEX
            logger.info(f"Fallback to camera index {FALLBACK_CAMERA_INDEX}")
            return True
        
        logger.error("Could not open any webcam")
        return False
    
    def _open_camera(self, index: int) -> bool:
        """Attempt to open a specific camera index."""
        self.cap = cv2.VideoCapture(index)
        return self.cap is not None and self.cap.isOpened()
    
    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Read a frame from the camera.
        
        Returns:
            Tuple of (success, frame). If success is False, frame is None.
        """
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to capture frame")
            return False, None
        
        return True, frame
    
    def flip_frame(self, frame: cv2.Mat, horizontal: bool = True) -> cv2.Mat:
        """
        Flip a frame horizontally (mirror effect).
        
        Args:
            frame: Input frame
            horizontal: Whether to flip horizontally
            
        Returns:
            Flipped frame
        """
        if horizontal:
            return cv2.flip(frame, 1)
        return frame
    
    def get_frame_shape(self, frame: cv2.Mat) -> Tuple[int, int, int]:
        """Get the shape of a frame (height, width, channels)."""
        return frame.shape
    
    def release(self):
        """Release the camera resource."""
        if self.cap is not None:
            self.cap.release()
            logger.info("Camera released")
    
    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self.cap is not None and self.cap.isOpened()
