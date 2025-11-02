# steering.py
import cv2
import mediapipe as mp
from pynput.keyboard import Controller
import math

# ==================== CONFIGURATION ====================
STEERING_SENSITIVITY = 2.0  # Higher = more sensitive steering
MOVEMENT_KEYS = ['w', 'a', 's', 'd']

# ==================== SETUP ====================
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
font = cv2.FONT_HERSHEY_SIMPLEX

keyboard = Controller()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


# ==================== HELPER FUNCTIONS ====================

def get_wrist_coords(hand_landmarks, w, h):
    """Extracts the wrist coordinates in pixel values."""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    return int(wrist.x * w), int(wrist.y * h)


def release_all_keys(keys=MOVEMENT_KEYS):
    """Releases all specified keyboard keys."""
    for key in keys:
        keyboard.release(key)


def calculate_steering_wheel_angle(hand1, hand2, screen_width):
    """
    Calculate steering angle like a steering wheel:
    - When hands rotate clockwise (right hand moves down, left hand moves up) -> Turn RIGHT
    - When hands rotate counter-clockwise (left hand moves down, right hand moves up) -> Turn LEFT
    """
    h1_x, h1_y = hand1
    h2_x, h2_y = hand2

    # Calculate the angle between the two hands
    dx = h2_x - h1_x
    dy = h2_y - h1_y

    # Calculate angle in radians, then convert to degrees
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    # Normalize angle to -180 to 180 range
    if angle_deg > 180:
        angle_deg -= 360
    elif angle_deg < -180:
        angle_deg += 360

    return angle_deg


def calculate_hand_tilt(hand1, hand2):
    """
    Calculate if hands are tilted like a steering wheel
    Returns: -1 (left turn), 0 (straight), 1 (right turn)
    """
    h1_x, h1_y = hand1
    h2_x, h2_y = hand2

    # Calculate the slope between hands
    if abs(h2_x - h1_x) > 10:  # Avoid division by zero
        slope = (h2_y - h1_y) / (h2_x - h1_x)

        # If slope is positive and steep, it's a right turn gesture
        if slope > 0.3:  # Right hand lower than left hand
            return 1  # Turn right
        # If slope is negative and steep, it's a left turn gesture
        elif slope < -0.3:  # Left hand lower than right hand
            return -1  # Turn left

    return 0  # Straight


# ==================== PROCESS CAMERA ====================

with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
) as hands:
    try:
        while True:
            success, frame = cap.read()
            if not success:
                continue

            # Flip for mirror view and get dimensions
            flipped_frame = cv2.flip(frame, 1)
            h, w, _ = flipped_frame.shape

            # Process with MediaPipe (convert to RGB)
            rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Core Hand Tracking and Coordinate Extraction
            coords = []
            handedness = []
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw landmarks on the flipped frame
                    mp_drawing.draw_landmarks(flipped_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Store wrist coordinates
                    coords.append(get_wrist_coords(hand_landmarks, w, h))

                    # Store handedness
                    if results.multi_handedness and idx < len(results.multi_handedness):
                        handedness.append(results.multi_handedness[idx].classification[0].label)
                    else:
                        handedness.append("Unknown")

            # Reset keys and determine state text
            release_all_keys()
            text = "No Hands Detected"
            debug_text = ""
            steering_angle = 0

            # ==================== STEERING WHEEL LOGIC ====================

            if len(coords) == 2:
                # Map hands to their actual positions
                left_hand_pos = None
                right_hand_pos = None

                for i, hand_label in enumerate(handedness):
                    if hand_label == 'Left':
                        left_hand_pos = coords[i]
                    elif hand_label == 'Right':
                        right_hand_pos = coords[i]

                if left_hand_pos and right_hand_pos:
                    left_x, left_y = left_hand_pos
                    right_x, right_y = right_hand_pos

                    # Calculate hand distance (for braking)
                    hand_distance = math.sqrt((left_x - right_x) ** 2 + (left_y - right_y) ** 2)

                    # STEERING WHEEL GESTURES:

                    # 1. BRAKING: Hands very close together
                    if hand_distance < 100:
                        text = "BRAKING 🛑"
                        keyboard.press('s')
                        debug_text = f"Hand Distance: {hand_distance:.0f}"

                    else:
                        # Always move forward when driving
                        keyboard.press('w')

                        # 2. STEERING WHEEL TILT DETECTION
                        tilt_direction = calculate_hand_tilt(left_hand_pos, right_hand_pos)

                        if tilt_direction == 1:
                            # Right turn: Right hand lower than left hand
                            text = "TURN RIGHT ➡️"
                            keyboard.press('d')
                            steering_angle = 45
                            debug_text = "Steering: RIGHT (Right hand lower)"

                        elif tilt_direction == -1:
                            # Left turn: Left hand lower than right hand
                            text = "TURN LEFT ⬅️"
                            keyboard.press('a')
                            steering_angle = -45
                            debug_text = "Steering: LEFT (Left hand lower)"

                        else:
                            # Straight: Hands are level
                            text = "GO STRAIGHT ⬆️"
                            steering_angle = 0
                            debug_text = "Steering: STRAIGHT (Hands level)"

                        # Additional debug info
                        debug_text += f" | Dist: {hand_distance:.0f}"

            elif len(coords) == 1:
                # Single hand detected - Reverse
                text = "REVERSE ⬇️"
                keyboard.press('s')
                debug_text = "Single hand detected"

            else:
                # No hands detected
                text = "NO HANDS - STOPPED 🛑"
                debug_text = "Show both hands to drive"

            # ==================== DISPLAY ====================

            # Main status
            cv2.putText(flipped_frame, text, (30, 50), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(flipped_frame, debug_text, (30, 90), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(flipped_frame, f"Hands detected: {len(coords)}", (30, 120), font, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)

            # Draw visual steering wheel
            center_x, center_y = w // 2, h // 2
            wheel_radius = 100

            # Draw steering wheel base
            cv2.circle(flipped_frame, (center_x, center_y), wheel_radius, (100, 100, 100), 2)

            # Draw steering wheel indicator based on steering angle
            if steering_angle != 0:
                indicator_angle = math.radians(steering_angle)
                end_x = center_x + int(wheel_radius * math.sin(indicator_angle))
                end_y = center_y - int(wheel_radius * math.cos(indicator_angle))
                cv2.line(flipped_frame, (center_x, center_y), (end_x, end_y), (0, 255, 0), 3)

            # Draw center line
            cv2.line(flipped_frame, (center_x, 0), (center_x, h), (50, 50, 50), 1)

            # Draw hand positions with connection line
            for i, (x, y) in enumerate(coords):
                if i < len(handedness):
                    color = (0, 0, 255) if handedness[i] == 'Left' else (255, 0, 0)
                    label = handedness[i]
                else:
                    color = (0, 255, 255)
                    label = f"Hand {i + 1}"

                cv2.circle(flipped_frame, (x, y), 12, color, -1)
                cv2.putText(flipped_frame, label, (x - 20, y - 15), font, 0.5, color, 1, cv2.LINE_AA)

            # Draw line connecting hands if both are detected
            if len(coords) == 2:
                cv2.line(flipped_frame, coords[0], coords[1], (255, 255, 0), 2)

                # Calculate and display the angle between hands
                angle = calculate_steering_wheel_angle(coords[0], coords[1], w)
                angle_text = f"Angle: {angle:.1f}°"
                cv2.putText(flipped_frame, angle_text, (w - 150, 50), font, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

            # Instructions
            cv2.putText(flipped_frame, "STEERING WHEEL GESTURES:", (30, h - 100), font, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(flipped_frame, "Right Turn: Move RIGHT hand DOWN", (30, h - 80), font, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(flipped_frame, "Left Turn: Move LEFT hand DOWN", (30, h - 60), font, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(flipped_frame, "Brake: Bring hands CLOSE together", (30, h - 40), font, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(flipped_frame, "Press 'q' to quit", (30, h - 15), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Virtual Steering Wheel Control", flipped_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Ensure all keys and resources are released upon exit
        release_all_keys()
        cap.release()
        cv2.destroyAllWindows()