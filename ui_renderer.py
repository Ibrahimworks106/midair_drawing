"""UI rendering for Air Drawing application."""

from typing import Tuple, Optional
from datetime import datetime
import cv2
import numpy as np

from config import (
    COLORS_LIST, TOOLBAR_HEIGHT, COLOR_CIRCLE_RADIUS,
    COLOR_CIRCLE_BORDER_RADIUS, COLOR_CIRCLE_Y, COLOR_SPACING_FACTOR,
    SELECTION_THRESHOLD, MODE_COLORS, DEFAULT_MODE, WHITE, BLACK,
    GRAY_LIGHT, GRAY_MEDIUM, GRAY_DARK, INSTRUCTIONS_TEXT,
    FONT_FACE, FONT_SCALE_MODE, FONT_SCALE_INSTRUCTIONS, FONT_SCALE_FPS,
    FONT_THICKNESS_MODE, FONT_THICKNESS_INSTRUCTIONS, FONT_THICKNESS_FPS,
    MODE_INDICATOR_X1, MODE_INDICATOR_Y1_OFFSET, MODE_INDICATOR_X2,
    MODE_INDICATOR_Y2_OFFSET, MODE_INDICATOR_TEXT_X, MODE_INDICATOR_TEXT_Y_OFFSET,
    TOOLBAR_OVERLAY_ALPHA, INSTRUCTIONS_OVERLAY_ALPHA
)


class UIRenderer:
    """Handles all UI rendering operations."""
    
    def __init__(self):
        self.save_msg_timer: float = 0
        self.mode: str = DEFAULT_MODE
    
    def set_mode(self, mode: str):
        """Set the current drawing mode."""
        self.mode = mode
    
    def draw_toolbar(
        self, 
        frame: np.ndarray,
        active_color: Tuple[int, int, int],
        active_thickness: int,
        width: int
    ) -> np.ndarray:
        """
        Draw the color picker toolbar at the top of the frame.
        
        Args:
            frame: The video frame
            active_color: Currently selected color
            active_thickness: Current brush thickness
            width: Frame width
            
        Returns:
            Frame with toolbar drawn
        """
        # Draw semi-transparent toolbar background
        overlay = frame.copy()
        cv2.rectangle(
            overlay, 
            (0, 0), 
            (width, TOOLBAR_HEIGHT), 
            GRAY_DARK, 
            -1
        )
        frame = cv2.addWeighted(overlay, TOOLBAR_OVERLAY_ALPHA, frame, 1 - TOOLBAR_OVERLAY_ALPHA, 0)
        
        # Draw color circles
        for i, (label, color) in enumerate(COLORS_LIST):
            cx = width // COLOR_SPACING_FACTOR * (i + 1)
            
            # Draw circle
            cv2.circle(frame, (cx, COLOR_CIRCLE_Y), COLOR_CIRCLE_RADIUS, color, -1)
            
            # Draw white border if active
            is_active = (
                color == active_color or 
                (label == "ERASE" and active_thickness > 10)
            )
            if is_active:
                cv2.circle(
                    frame, 
                    (cx, COLOR_CIRCLE_Y), 
                    COLOR_CIRCLE_BORDER_RADIUS, 
                    WHITE, 
                    2
                )
        
        return frame
    
    def check_color_selection(
        self, 
        x: int, 
        y: int,
        width: int
    ) -> Optional[Tuple[str, Tuple[int, int, int], bool]]:
        """
        Check if a color circle was selected.
        
        Args:
            x: Cursor x coordinate
            y: Cursor y coordinate
            width: Frame width
            
        Returns:
            Tuple of (label, color, is_eraser) if selected, None otherwise
        """
        if y >= TOOLBAR_HEIGHT:
            return None
        
        for i, (label, color) in enumerate(COLORS_LIST):
            cx = width // COLOR_SPACING_FACTOR * (i + 1)
            if abs(x - cx) < SELECTION_THRESHOLD:
                is_eraser = (label == "ERASE")
                return (label, color, is_eraser)
        
        return None
    
    def draw_mode_indicator(
        self, 
        frame: np.ndarray,
        height: int
    ) -> np.ndarray:
        """
        Draw the mode indicator pill at bottom left.
        
        Args:
            frame: The video frame
            height: Frame height
            
        Returns:
            Frame with mode indicator drawn
        """
        pill_color = MODE_COLORS.get(self.mode, MODE_COLORS["DRAW"])
        
        # Draw pill background
        cv2.rectangle(
            frame,
            (MODE_INDICATOR_X1, height - MODE_INDICATOR_Y1_OFFSET),
            (MODE_INDICATOR_X2, height - MODE_INDICATOR_Y2_OFFSET),
            pill_color,
            -1
        )
        cv2.rectangle(
            frame,
            (MODE_INDICATOR_X1, height - MODE_INDICATOR_Y1_OFFSET),
            (MODE_INDICATOR_X2, height - MODE_INDICATOR_Y2_OFFSET),
            WHITE,
            1
        )
        
        # Draw mode text
        cv2.putText(
            frame,
            f"MODE: {self.mode}",
            (MODE_INDICATOR_TEXT_X, height - MODE_INDICATOR_TEXT_Y_OFFSET),
            FONT_FACE,
            FONT_SCALE_MODE,
            BLACK,
            FONT_THICKNESS_MODE
        )
        
        return frame
    
    def draw_active_color_indicator(
        self,
        frame: np.ndarray,
        active_color: Tuple[int, int, int],
        height: int
    ) -> np.ndarray:
        """
        Draw the active color indicator circle.
        
        Args:
            frame: The video frame
            active_color: Currently selected color
            height: Frame height
            
        Returns:
            Frame with color indicator drawn
        """
        cv2.circle(frame, (210, height - 40), 15, active_color, -1)
        cv2.circle(frame, (210, height - 40), 15, WHITE, 1)
        return frame
    
    def draw_instructions_bar(
        self,
        frame: np.ndarray,
        height: int
    ) -> np.ndarray:
        """
        Draw the instructions bar at the bottom.
        
        Args:
            frame: The video frame
            height: Frame height
            
        Returns:
            Frame with instructions bar drawn
        """
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, height - 25), (frame.shape[1], height), BLACK, -1)
        frame = cv2.addWeighted(overlay, INSTRUCTIONS_OVERLAY_ALPHA, frame, 1 - INSTRUCTIONS_OVERLAY_ALPHA, 0)
        
        cv2.putText(
            frame,
            INSTRUCTIONS_TEXT,
            (10, height - 8),
            FONT_FACE,
            FONT_SCALE_INSTRUCTIONS,
            GRAY_LIGHT,
            FONT_THICKNESS_INSTRUCTIONS
        )
        
        return frame
    
    def draw_fps_counter(
        self,
        frame: np.ndarray,
        fps: float,
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Draw the FPS counter at top right.
        
        Args:
            frame: The video frame
            fps: Current FPS value
            width: Frame width
            height: Frame height
            
        Returns:
            Frame with FPS counter drawn
        """
        cv2.putText(
            frame,
            f"FPS: {fps:.0f}",
            (width - 90, height - 33),
            FONT_FACE,
            FONT_SCALE_FPS,
            GRAY_MEDIUM,
            FONT_THICKNESS_FPS
        )
        return frame
    
    def draw_save_confirmation(
        self,
        frame: np.ndarray,
        current_time: float,
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Draw save confirmation message if recently saved.
        
        Args:
            frame: The video frame
            current_time: Current timestamp
            width: Frame width
            height: Frame height
            
        Returns:
            Frame with save message if applicable
        """
        from config import SAVE_MESSAGE_DURATION
        
        if current_time - self.save_msg_timer < SAVE_MESSAGE_DURATION:
            cv2.putText(
                frame,
                "Saved!",
                (width // 2 - 60, height // 2),
                FONT_FACE,
                1.5,
                (0, 255, 0),
                3
            )
        return frame
    
    def trigger_save_message(self):
        """Trigger the save confirmation message."""
        import time
        self.save_msg_timer = time.time()
    
    def draw_fingertip_marker(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Draw a marker at the fingertip position.
        
        Args:
            frame: The video frame
            x: X coordinate
            y: Y coordinate
            color: Marker color
            
        Returns:
            Frame with marker drawn
        """
        cv2.circle(frame, (x, y), 12, color, -1)
        return frame
    
    def create_splash_screen(
        self,
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Create a splash screen image.
        
        Args:
            width: Frame width
            height: Frame height
            
        Returns:
            Splash screen image
        """
        splash = np.zeros((height, width, 3), dtype=np.uint8)
        
        cv2.putText(
            splash,
            "MID-AIR DRAWING",
            (width // 2 - 220, height // 2 - 20),
            FONT_FACE,
            1.8,
            (0, 255, 0),
            3
        )
        cv2.putText(
            splash,
            "Show your hand to begin",
            (width // 2 - 200, height // 2 + 40),
            FONT_FACE,
            1.0,
            WHITE,
            2
        )
        
        return splash
    
    def save_canvas(
        self,
        canvas: np.ndarray
    ) -> str:
        """
        Save the canvas to a file.
        
        Args:
            canvas: The canvas to save
            
        Returns:
            Filename of the saved image
        """
        filename = f"drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, canvas)
        return filename
