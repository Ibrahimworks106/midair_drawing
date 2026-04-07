import cv2
import mediapipe as mp
import numpy as np
import time
import os
from datetime import datetime
from collections import deque

# Import the new Tasks API components
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def is_finger_up(hand_landmarks, tip_id, pip_id):
    """Returns True if the finger is extended (tip y < pip y)."""
    return hand_landmarks[tip_id].y < hand_landmarks[pip_id].y


def main():
    # OBS Virtual Camera Index
    camera_index = 1
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {camera_index}")
        # Try a fallback index
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open any webcam.")
            return
        print("Fallback to camera index 0")
    else:
        print(f"Camera {camera_index} opened successfully")

    # Initialize MediaPipe Hand Landmarker
    # Note: Modern MediaPipe (v0.10+) uses the Tasks API.
    # We need a model bundle. If not found, we'll try to handle it gracefully.
    model_path = 'hand_landmarker.task'
    
    # Check if model exists, if not, we might have issues
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found. Hand detection may fail.")
        print("Please download it from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO
    )
    
    detector = None
    try:
        detector = vision.HandLandmarker.create_from_options(options)
    except Exception as e:
        print(f"Error initializing MediaPipe detector: {e}")
        print("This might be due to a missing 'hand_landmarker.task' file or environment issues.")
        # Fallback to legacy if possible (rarely works if Tasks API is the only one present)
        try:
            from mediapipe.python.solutions import hands as mp_hands
            from mediapipe.python.solutions import drawing_utils as mp_drawing
            detector_legacy = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
            print("Successfully fell back to legacy MediaPipe solutions.")
        except:
            return

    # Use the drawing utils from the tasks or legacy
    try:
        from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
    except ImportError:
        try:
            from mediapipe.python.solutions import drawing_utils as mp_drawing
        except ImportError:
            mp_drawing = None

    # Initialize canvas and previous point for persistent drawing
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read initial frame for canvas")
        return
    canvas = np.zeros_like(frame)
    prev_point = None
    smooth_buffer = deque(maxlen=5)
    h, w, _ = frame.shape

    # Color picker toolbar configuration - circle-based system
    colors_list = [
        ("RED", (0, 0, 255)), ("BLUE", (255, 0, 0)), ("GREEN", (0, 255, 0)),
        ("YELLOW", (0, 255, 255)), ("WHITE", (255, 255, 255)), ("ERASE", (80, 80, 80))
    ]
    active_color = (0, 255, 0)  # Default green
    active_thickness = 5

    # FPS calculation
    fps = 0
    last_time = time.time()

    # Create a window for display
    cv2.namedWindow("Air Drawing", cv2.WINDOW_NORMAL)

    # Initialize mode
    mode = "STOP"  # Default mode
    save_msg_timer = 0
    clear_start_time = None

    # Splash screen
    splash = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(splash, "MID-AIR DRAWING", (w//2 - 220, h//2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3)
    cv2.putText(splash, "Show your hand to begin", (w//2 - 200, h//2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imshow("Air Drawing", splash)
    cv2.waitKey(2000)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        frame = cv2.flip(frame, 1)

        # New API processing
        if detector:
            # Convert frame from BGR to RGB before creating MediaPipe Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Detect hand landmarks (VIDEO mode requires timestamp in milliseconds)
            detection_result = detector.detect_for_video(mp_image, int(time.time() * 1000))

            # Draw landmarks if detected
            if detection_result.hand_landmarks:
                h, w, _ = frame.shape
                for hand_landmarks in detection_result.hand_landmarks:
                    # Check which fingers are up for mode control
                    index_up  = is_finger_up(hand_landmarks, 8, 6)   # INDEX_FINGER_TIP, INDEX_FINGER_PIP
                    middle_up = is_finger_up(hand_landmarks, 12, 10) # MIDDLE_FINGER_TIP, MIDDLE_FINGER_PIP
                    ring_up   = is_finger_up(hand_landmarks, 16, 14) # RING_FINGER_TIP, RING_FINGER_DIP
                    pinky_up  = is_finger_up(hand_landmarks, 20, 18) # PINKY_TIP, PINKY_DIP

                    # Set mode based on finger combination
                    if index_up and not middle_up and not ring_up and not pinky_up:
                        mode = "DRAW"
                    elif index_up and middle_up and not ring_up and not pinky_up:
                        mode = "STOP"
                    elif index_up and middle_up and ring_up and pinky_up:
                        if clear_start_time is None:
                            clear_start_time = time.time()
                        elif time.time() - clear_start_time > 1.0:
                            mode = "CLEAR"
                            canvas = np.zeros_like(frame)
                            clear_start_time = None
                    else:
                        clear_start_time = None

                    # Extract index fingertip (Landmark 8)
                    landmark = hand_landmarks[8]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    # Simple fixed smoothing
                    smooth_buffer.append((x, y))
                    sx = int(np.mean([p[0] for p in smooth_buffer]))
                    sy = int(np.mean([p[1] for p in smooth_buffer]))

                    # Check if fingertip is in toolbar region (y < 70)
                    if y < 70:
                        # Do NOT draw on canvas when in toolbar
                        prev_point = None
                        # Check which color circle was selected
                        for i, (label, color) in enumerate(colors_list):
                            cx = w // 7 * (i + 1)
                            if abs(x - cx) < 30:
                                if label == "ERASE":
                                    active_color = (0, 0, 0)
                                    active_thickness = 30
                                else:
                                    active_color = color
                                    active_thickness = 5
                                break
                    else:
                        # Only draw on canvas when in DRAW mode and not in toolbar
                        if mode == "DRAW":
                            # Draw line on canvas if previous point exists and finger moved > 3px
                            if prev_point is not None:
                                dist = np.hypot(sx - prev_point[0], sy - prev_point[1])
                                if dist > 3:
                                    cv2.line(canvas, prev_point, (sx, sy), active_color, thickness=active_thickness)
                                    prev_point = (sx, sy)
                            else:
                                prev_point = (sx, sy)
                        else:
                            prev_point = None

                    # Draw filled circle at smoothed fingertip position
                    cv2.circle(frame, (sx, sy), 12, active_color, -1)
            else:
                # No hand detected, reset prev_point to prevent jump lines
                prev_point = None
        elif 'detector_legacy' in locals():
            # Legacy processing fallback
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector_legacy.process(frame_rgb)
            if results.multi_hand_landmarks:
                h, w, _ = frame.shape
                for hand_landmarks in results.multi_hand_landmarks:
                    # Check which fingers are up for mode control
                    index_up  = is_finger_up(hand_landmarks, 8, 6)
                    middle_up = is_finger_up(hand_landmarks, 12, 10)
                    ring_up   = is_finger_up(hand_landmarks, 16, 14)
                    pinky_up  = is_finger_up(hand_landmarks, 20, 18)

                    # Set mode based on finger combination
                    if index_up and not middle_up and not ring_up and not pinky_up:
                        mode = "DRAW"
                    elif index_up and middle_up and not ring_up and not pinky_up:
                        mode = "STOP"
                    elif index_up and middle_up and ring_up and pinky_up:
                        if clear_start_time is None:
                            clear_start_time = time.time()
                        elif time.time() - clear_start_time > 1.0:
                            mode = "CLEAR"
                            canvas = np.zeros_like(frame)
                            clear_start_time = None
                    else:
                        clear_start_time = None

                    # Extract index fingertip (Landmark 8)
                    landmark = hand_landmarks[8]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    # Simple fixed smoothing
                    smooth_buffer.append((x, y))
                    sx = int(np.mean([p[0] for p in smooth_buffer]))
                    sy = int(np.mean([p[1] for p in smooth_buffer]))

                    # Check if fingertip is in toolbar region (y < 70)
                    if y < 70:
                        prev_point = None
                        for i, (label, color) in enumerate(colors_list):
                            cx = w // 7 * (i + 1)
                            if abs(x - cx) < 30:
                                if label == "ERASE":
                                    active_color = (0, 0, 0)
                                    active_thickness = 30
                                else:
                                    active_color = color
                                    active_thickness = 5
                                break
                    else:
                        if mode == "DRAW":
                            if prev_point is not None:
                                dist = np.hypot(sx - prev_point[0], sy - prev_point[1])
                                if dist > 3:
                                    cv2.line(canvas, prev_point, (sx, sy), active_color, thickness=active_thickness)
                                    prev_point = (sx, sy)
                            else:
                                prev_point = (sx, sy)
                        else:
                            prev_point = None

                    # Draw filled circle at smoothed fingertip position
                    cv2.circle(frame, (sx, sy), 12, active_color, -1)
            else:
                prev_point = None

        # Blend canvas onto the frame
        output = cv2.addWeighted(frame, 1.0, canvas, 0.8, 0)

        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - last_time)
        last_time = current_time

        # Draw semi-transparent toolbar background
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (20, 20, 20), -1)
        output = cv2.addWeighted(overlay, 0.7, output, 0.3, 0)

        # Draw color circles
        circle_y = 35
        for i, (label, color) in enumerate(colors_list):
            cx = w // 7 * (i + 1)
            # Draw circle
            cv2.circle(output, (cx, circle_y), 22, color, -1)
            # Draw white border if active
            if color == active_color or (label == "ERASE" and active_thickness == 30):
                cv2.circle(output, (cx, circle_y), 25, (255, 255, 255), 2)

        # Display mode label with pill background
        mode_colors = {"DRAW": (0, 200, 80), "STOP": (0, 200, 220), "CLEAR": (220, 60, 60)}
        pill_color = mode_colors.get(mode, (0, 200, 80))
        cv2.rectangle(output, (15, h - 55), (180, h - 25), pill_color, -1)
        cv2.rectangle(output, (15, h - 55), (180, h - 25), (255, 255, 255), 1)
        cv2.putText(output, f"MODE: {mode}", (22, h - 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Active color indicator
        cv2.circle(output, (210, h - 40), 15, active_color, -1)
        cv2.circle(output, (210, h - 40), 15, (255, 255, 255), 1)

        # Draw semi-transparent instructions bar at bottom
        overlay2 = output.copy()
        cv2.rectangle(overlay2, (0, h - 25), (w, h), (0, 0, 0), -1)
        output = cv2.addWeighted(overlay2, 0.6, output, 0.4, 0)
        cv2.putText(output, "1 finger=Draw  |  2 fingers=Stop  |  Palm=Clear  |  S=Save  |  Q=Quit",
                    (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # FPS counter - top right
        cv2.putText(output, f"FPS: {fps:.0f}", (w - 90, h - 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Save confirmation message
        if time.time() - save_msg_timer < 2:
            cv2.putText(output, "Saved!", (w // 2 - 60, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow("Air Drawing", output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(filename, canvas)
            save_msg_timer = time.time()

    cap.release()
    cv2.destroyAllWindows()
    if detector:
        detector.close()

if __name__ == "__main__":
    main()
