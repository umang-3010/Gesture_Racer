import cv2
import mediapipe as mp
import ctypes
import sys

# ----------------- WINDOWS KEY CONTROL SETUP -----------------
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_KEYUP = 0x0002
INPUT_KEYBOARD = 1

# --- Key scan codes ---
KEYS = {
    "w": 0x11,  # Forward
    "a": 0x1E,  # Left
    "s": 0x1F,  # Backward
    "d": 0x20,  # Right
}

# --- ctypes structures ---
PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]


def _send_input(scan_code, flags):
    """Internal: Sends a low-level keyboard event"""
    extra = ctypes.c_ulong(0)
    ki = KeyBdInput(0, scan_code, flags, 0, ctypes.pointer(extra))
    ii_ = Input_I()
    ii_.ki = ki
    command = Input(ctypes.c_ulong(INPUT_KEYBOARD), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))


def press_key(key: str):
    """Press and hold a key"""
    if key.lower() in KEYS:
        _send_input(KEYS[key.lower()], KEYEVENTF_SCANCODE)


def release_key(key: str):
    """Release a pressed key"""
    if key.lower() in KEYS:
        _send_input(KEYS[key.lower()], KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP)


def release_all():
    """Release all movement keys"""
    for k in KEYS:
        release_key(k)


# ----------------- CONFIGURATION -----------------
TURN_THRESHOLD = 50  # Pixels from center to trigger turn

# ----------------- MEDIAPIPE & OPENCV SETUP -----------------
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    release_all()
    sys.exit()


# ----------------- HELPER FUNCTIONS -----------------

def get_wrist_coords(hand_landmarks, w, h):
    """Extracts the wrist coordinates in pixel values."""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    return int(wrist.x * w), int(wrist.y * h)


# ----------------- PROCESS CAMERA -----------------

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

            # Flip for mirror view
            flipped_frame = cv2.flip(frame, 1)
            h, w, _ = flipped_frame.shape

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            coords = []
            handedness = []

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(flipped_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    coords.append(get_wrist_coords(hand_landmarks, w, h))

                    if results.multi_handedness and idx < len(results.multi_handedness):
                        handedness.append(results.multi_handedness[idx].classification[0].label)
                    else:
                        handedness.append("Unknown")

            # Release all keys at start of each frame
            release_all()
            text = "No Hands - STOPPED"
            debug_text = ""

            # ==================== SIMPLIFIED STEERING LOGIC ====================

            if len(coords) == 2:
                # Calculate average position of both hands (steering center)
                avg_x = sum(x for x, y in coords) // len(coords)
                avg_y = sum(y for x, y in coords) // len(coords)

                # Calculate hand spread
                hand_spread = abs(coords[0][0] - coords[1][0])

                debug_text = f"Center: {avg_x}, Spread: {hand_spread}"

                # STEERING LOGIC:
                if hand_spread < 80:  # Hands close together = brake
                    text = "BRAKING 🛑"
                    press_key('s')
                else:  # Hands apart = drive
                    press_key('w')  # Always move forward

                    if avg_x < w // 2 - TURN_THRESHOLD:
                        text = "TURN LEFT ⬅️"
                        press_key('a')
                    elif avg_x > w // 2 + TURN_THRESHOLD:
                        text = "TURN RIGHT ➡️"
                        press_key('d')
                    else:
                        text = "GO STRAIGHT ⬆️"

            elif len(coords) == 1:
                # Single hand = reverse
                text = "REVERSE ⬇️"
                press_key('s')
                debug_text = "Single hand detected"

            else:
                # No hands = stop
                text = "NO HANDS - STOPPED 🛑"
                debug_text = "Show both hands to drive"

            # ==================== DISPLAY ====================
            cv2.putText(flipped_frame, text, (30, 50), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(flipped_frame, debug_text, (30, 90), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(flipped_frame, f"Hands: {len(coords)}", (30, 120), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw center line
            cv2.line(flipped_frame, (w // 2, 0), (w // 2, h), (100, 100, 100), 1)
            # Draw turn thresholds
            cv2.line(flipped_frame, (w // 2 - TURN_THRESHOLD, 0), (w // 2 - TURN_THRESHOLD, h), (50, 50, 50), 1)
            cv2.line(flipped_frame, (w // 2 + TURN_THRESHOLD, 0), (w // 2 + TURN_THRESHOLD, h), (50, 50, 50), 1)

            # Draw hand positions
            for i, (x, y) in enumerate(coords):
                color = (0, 0, 255) if i < len(handedness) and handedness[i] == 'Left' else (255, 0, 0)
                cv2.circle(flipped_frame, (x, y), 10, color, -1)
                if i < len(handedness):
                    cv2.putText(flipped_frame, handedness[i], (x - 20, y - 15), font, 0.5, color, 1, cv2.LINE_AA)

            cv2.putText(flipped_frame, "Press 'q' to quit", (w - 200, h - 20), font, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.imshow("Virtual Steering Wheel", flipped_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        release_all()
        cap.release()
        cv2.destroyAllWindows()