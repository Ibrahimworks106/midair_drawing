"""Hand detection service using MediaPipe."""

from typing import Optional, Any
import cv2
import os
import logging

from config import (
    MODEL_PATH, MODEL_URL,
    MIN_HAND_DETECTION_CONFIDENCE, MIN_HAND_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE, NUM_HANDS
)

logger = logging.getLogger(__name__)


class HandDetectionService:
    """Service for hand detection using MediaPipe."""
    
    def __init__(self):
        self.detector: Optional[Any] = None
        self.detector_legacy: Optional[Any] = None
        self.mp_drawing: Optional[Any] = None
        self.is_legacy: bool = False
    
    def initialize(self) -> bool:
        """
        Initialize the hand detector.
        
        Returns:
            True if initialization successful, False otherwise
        """
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"{MODEL_PATH} not found.")
            logger.warning(f"Please download it from: {MODEL_URL}")
        
        try:
            return self._initialize_modern_api()
        except Exception as e:
            logger.error(f"Error initializing modern MediaPipe API: {e}")
            logger.info("Attempting fallback to legacy API...")
            
            try:
                return self._initialize_legacy_api()
            except Exception as e2:
                logger.error(f"Legacy API initialization also failed: {e2}")
                return False
    
    def _initialize_modern_api(self) -> bool:
        """Initialize using the modern Tasks API."""
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=NUM_HANDS,
            min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            running_mode=vision.RunningMode.VIDEO
        )
        
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Try to import drawing utils from tasks API
        try:
            from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
            self.mp_drawing = mp_drawing
        except ImportError:
            logger.warning("Could not import drawing utils from tasks API")
        
        self.is_legacy = False
        logger.info("Modern MediaPipe API initialized successfully")
        return True
    
    def _initialize_legacy_api(self) -> bool:
        """Initialize using the legacy MediaPipe solutions API."""
        from mediapipe.python.solutions import hands as mp_hands
        from mediapipe.python.solutions import drawing_utils as mp_drawing
        
        self.detector_legacy = mp_hands.Hands(
            max_num_hands=NUM_HANDS,
            min_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE
        )
        self.mp_drawing = mp_drawing
        self.is_legacy = True
        
        logger.info("Legacy MediaPipe API initialized successfully")
        return True
    
    def detect_landmarks(
        self, 
        frame: Any,
        timestamp_ms: Optional[int] = None
    ):
        """
        Detect hand landmarks in a frame.
        
        Args:
            frame: Input frame (BGR format)
            timestamp_ms: Timestamp in milliseconds (for VIDEO mode)
            
        Returns:
            Detection results object or None if detection failed
        """
        if self.detector and not self.is_legacy:
            return self._detect_modern(frame, timestamp_ms)
        elif self.detector_legacy:
            return self._detect_legacy(frame)
        else:
            logger.error("No detector initialized")
            return None
    
    def _detect_modern(self, frame: Any, timestamp_ms: Optional[int] = None):
        """Detect using modern API."""
        import mediapipe as mp
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        if timestamp_ms is None:
            import time
            timestamp_ms = int(time.time() * 1000)
        
        return self.detector.detect_for_video(mp_image, timestamp_ms)
    
    def _detect_legacy(self, frame: Any):
        """Detect using legacy API."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.detector_legacy.process(frame_rgb)
    
    def get_hand_landmarks(self, detection_result) -> list:
        """
        Extract hand landmarks from detection result.
        
        Args:
            detection_result: Result from detect_landmarks
            
        Returns:
            List of hand landmarks
        """
        if self.is_legacy:
            if detection_result and detection_result.multi_hand_landmarks:
                return detection_result.multi_hand_landmarks
            return []
        else:
            if detection_result and detection_result.hand_landmarks:
                return detection_result.hand_landmarks
            return []
    
    def close(self):
        """Clean up resources."""
        if self.detector:
            try:
                self.detector.close()
            except Exception as e:
                logger.error(f"Error closing detector: {e}")
        
        if self.detector_legacy:
            try:
                self.detector_legacy.close()
            except Exception as e:
                logger.error(f"Error closing legacy detector: {e}")
    
    def is_initialized(self) -> bool:
        """Check if detector is initialized."""
        return self.detector is not None or self.detector_legacy is not None
