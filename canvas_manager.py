"""Canvas management for Air Drawing application."""

from typing import Optional, Tuple
import numpy as np
import cv2

from config import (
    DEFAULT_DRAWING_THICKNESS, DEFAULT_ERASER_THICKNESS,
    DEFAULT_MIN_MOVEMENT_DISTANCE, DEFAULT_COLOR,
    CANVAS_ALPHA, FRAME_ALPHA
)


class DrawingState:
    """Manages the current drawing state including color and thickness."""
    
    def __init__(self):
        self.active_color: Tuple[int, int, int] = DEFAULT_COLOR
        self.active_thickness: int = DEFAULT_DRAWING_THICKNESS
    
    def set_color(self, color: Tuple[int, int, int], is_eraser: bool = False):
        """Set the active drawing color and thickness."""
        if is_eraser:
            self.active_color = (0, 0, 0)
            self.active_thickness = DEFAULT_ERASER_THICKNESS
        else:
            self.active_color = color
            self.active_thickness = DEFAULT_DRAWING_THICKNESS
    
    def reset(self):
        """Reset to default drawing state."""
        self.active_color = DEFAULT_COLOR
        self.active_thickness = DEFAULT_DRAWING_THICKNESS


class CanvasManager:
    """Manages the drawing canvas and drawing operations."""
    
    def __init__(self, frame_shape: Tuple[int, int, int]):
        """
        Initialize the canvas manager.
        
        Args:
            frame_shape: Shape of the video frame (height, width, channels)
        """
        self.canvas = np.zeros(frame_shape, dtype=np.uint8)
        self.drawing_state = DrawingState()
        self.prev_point: Optional[Tuple[int, int]] = None
    
    def clear_canvas(self):
        """Clear the entire canvas."""
        self.canvas.fill(0)
        self.prev_point = None
    
    def draw_point(
        self, 
        x: int, 
        y: int, 
        mode: str,
        min_distance: int = DEFAULT_MIN_MOVEMENT_DISTANCE
    ):
        """
        Draw a point on the canvas if in DRAW mode.
        
        Args:
            x: X coordinate
            y: Y coordinate
            mode: Current drawing mode ("DRAW", "STOP", or "CLEAR")
            min_distance: Minimum distance to move before drawing
        """
        current_point = (x, y)
        
        if mode == "DRAW":
            if self.prev_point is not None:
                dist = np.hypot(
                    current_point[0] - self.prev_point[0],
                    current_point[1] - self.prev_point[1]
                )
                if dist > min_distance:
                    cv2.line(
                        self.canvas,
                        self.prev_point,
                        current_point,
                        self.drawing_state.active_color,
                        thickness=self.drawing_state.active_thickness
                    )
                    self.prev_point = current_point
            else:
                self.prev_point = current_point
        else:
            self.prev_point = None
    
    def reset_point(self):
        """Reset the previous point (used when entering toolbar area)."""
        self.prev_point = None
    
    def get_canvas(self) -> np.ndarray:
        """Get the current canvas."""
        return self.canvas
    
    def blend_with_frame(
        self, 
        frame: np.ndarray,
        canvas_alpha: float = CANVAS_ALPHA,
        frame_alpha: float = FRAME_ALPHA
    ) -> np.ndarray:
        """
        Blend the canvas onto the frame.
        
        Args:
            frame: The video frame
            canvas_alpha: Alpha value for canvas (0-1)
            frame_alpha: Alpha value for frame (0-1)
            
        Returns:
            Blended output image
        """
        return cv2.addWeighted(frame, frame_alpha, self.canvas, canvas_alpha, 0)
    
    def update_canvas_size(self, frame_shape: Tuple[int, int, int]):
        """
        Update canvas size when frame dimensions change.
        
        Args:
            frame_shape: New frame shape (height, width, channels)
        """
        self.canvas = np.zeros(frame_shape, dtype=np.uint8)
        self.prev_point = None
