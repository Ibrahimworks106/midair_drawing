"""
Air Drawing Application - Main Entry Point

A gesture-controlled air drawing application using MediaPipe hand tracking.
Shows your hand to begin drawing in mid-air with color selection and eraser.

Usage:
    python main.py

Controls:
    - 1 finger (index): DRAW mode
    - 2 fingers (index + middle): STOP mode
    - Full palm (all fingers): CLEAR canvas (hold for 1 second)
    - S: Save current drawing
    - Q: Quit application
"""

import logging
import time
import cv2

from config import (
    WINDOW_NAME, SPLASH_DISPLAY_DURATION, DEFAULT_MODE
)
from camera_manager import CameraManager
from hand_detection_service import HandDetectionService
from gesture_detector import GestureDetector, LandmarkProcessor
from canvas_manager import CanvasManager
from ui_renderer import UIRenderer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AirDrawingApp:
    """Main application class for Air Drawing."""
    
    def __init__(self):
        self.camera_manager = CameraManager()
        self.detection_service = HandDetectionService()
        self.gesture_detector = GestureDetector()
        self.landmark_processor = LandmarkProcessor()
        self.canvas_manager = None
        self.ui_renderer = UIRenderer()
        
        self.fps = 0.0
        self.last_time = time.time()
        self.running = False
    
    def initialize(self) -> bool:
        """Initialize all application components."""
        logger.info("Initializing Air Drawing Application...")
        
        # Initialize camera
        if not self.camera_manager.open():
            logger.error("Failed to initialize camera")
            return False
        
        # Initialize hand detection
        if not self.detection_service.initialize():
            logger.error("Failed to initialize hand detection")
            self.camera_manager.release()
            return False
        
        # Read initial frame to get dimensions
        ret, frame = self.camera_manager.read()
        if not ret:
            logger.error("Failed to read initial frame")
            self.cleanup()
            return False
        
        # Flip frame for mirror effect
        frame = self.camera_manager.flip_frame(frame)
        
        # Initialize canvas and managers with frame dimensions
        frame_shape = self.camera_manager.get_frame_shape(frame)
        self.canvas_manager = CanvasManager(frame_shape)
        
        logger.info("Initialization complete")
        return True
    
    def show_splash_screen(self, width: int, height: int):
        """Display splash screen."""
        splash = self.ui_renderer.create_splash_screen(width, height)
        cv2.imshow(WINDOW_NAME, splash)
        cv2.waitKey(SPLASH_DISPLAY_DURATION)
    
    def process_frame(self, frame) -> bool:
        """
        Process a single frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            True to continue running, False to exit
        """
        # Flip frame horizontally for mirror effect
        frame = self.camera_manager.flip_frame(frame)
        height, width, _ = frame.shape
        
        # Detect hand landmarks
        detection_result = self.detection_service.detect_landmarks(frame)
        hand_landmarks_list = self.detection_service.get_hand_landmarks(detection_result)
        
        if hand_landmarks_list:
            # Process first detected hand
            hand_landmarks = hand_landmarks_list[0]
            
            # Detect gesture and update mode
            mode = self.gesture_detector.detect_gesture(hand_landmarks)
            self.ui_renderer.set_mode(mode)
            
            # Handle clear mode
            if mode == "CLEAR":
                self.canvas_manager.clear_canvas()
                self.gesture_detector.reset()
            
            # Extract and smooth fingertip coordinates
            x, y = self.landmark_processor.extract_fingertip(
                hand_landmarks, width, height
            )
            sx, sy = self.landmark_processor.smooth_coordinates(x, y)
            
            # Check for color selection in toolbar
            color_selection = self.ui_renderer.check_color_selection(sx, sy, width)
            
            if color_selection:
                # In toolbar region - select color
                label, color, is_eraser = color_selection
                self.canvas_manager.drawing_state.set_color(color, is_eraser)
                self.canvas_manager.reset_point()
            else:
                # Not in toolbar - draw on canvas
                self.canvas_manager.draw_point(sx, sy, mode)
            
            # Draw fingertip marker
            frame = self.ui_renderer.draw_fingertip_marker(
                frame, sx, sy, self.canvas_manager.drawing_state.active_color
            )
        else:
            # No hand detected - reset state
            self.landmark_processor.reset()
            self.canvas_manager.reset_point()
        
        # Blend canvas with frame
        output = self.canvas_manager.blend_with_frame(frame)
        
        # Draw UI elements
        output = self.ui_renderer.draw_toolbar(
            output,
            self.canvas_manager.drawing_state.active_color,
            self.canvas_manager.drawing_state.active_thickness,
            width
        )
        output = self.ui_renderer.draw_mode_indicator(output, height)
        output = self.ui_renderer.draw_active_color_indicator(
            output,
            self.canvas_manager.drawing_state.active_color,
            height
        )
        output = self.ui_renderer.draw_instructions_bar(output, height)
        
        # Calculate and draw FPS
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        output = self.ui_renderer.draw_fps_counter(output, self.fps, width, height)
        
        # Draw save confirmation if applicable
        output = self.ui_renderer.draw_save_confirmation(output, current_time, width, height)
        
        # Display the output
        cv2.imshow(WINDOW_NAME, output)
        
        # Handle keyboard input
        return self.handle_keyboard_input(output)
    
    def handle_keyboard_input(self, canvas) -> bool:
        """
        Handle keyboard input.
        
        Args:
            canvas: Current canvas for saving
            
        Returns:
            True to continue running, False to exit
        """
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return False
        elif key == ord('s'):
            filename = self.ui_renderer.save_canvas(canvas)
            logger.info(f"Drawing saved as: {filename}")
            self.ui_renderer.trigger_save_message()
        
        return True
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        self.camera_manager.release()
        self.detection_service.close()
        cv2.destroyAllWindows()
        logger.info("Cleanup complete")
    
    def run(self):
        """Main application loop."""
        if not self.initialize():
            return
        
        # Get frame dimensions for splash screen
        ret, frame = self.camera_manager.read()
        if ret:
            frame = self.camera_manager.flip_frame(frame)
            height, width, _ = frame.shape
            self.show_splash_screen(width, height)
        
        self.running = True
        logger.info("Starting main loop...")
        
        try:
            while self.running:
                ret, frame = self.camera_manager.read()
                if not ret:
                    logger.warning("Failed to read frame, continuing...")
                    continue
                
                self.running = self.process_frame(frame)
                
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        finally:
            self.cleanup()


def main():
    """Entry point for the Air Drawing application."""
    app = AirDrawingApp()
    app.run()


if __name__ == "__main__":
    main()
