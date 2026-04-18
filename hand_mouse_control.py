"""
Hand mouse controller for macOS.

Install dependencies:
    pip install opencv-python mediapipe pyobjc-framework-Quartz

On first run, the script downloads the MediaPipe hand landmarker model into
the local project folder.

On macOS, allow camera access and Accessibility control for Python/your IDE.
Press "q" in the preview window to quit.
"""

from pathlib import Path
import math
import time
import urllib.request

import cv2
import mediapipe as mp
import Quartz
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark


# =========================
# Tuning constants
# =========================
CAMERA_INDEX = 0
WINDOW_NAME = "Hand Mouse Control"
PREVIEW_WIDTH = 960
PREVIEW_HEIGHT = 720

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

NUM_HANDS = 1
MIN_HAND_DETECTION_CONFIDENCE = 0.50
MIN_HAND_PRESENCE_CONFIDENCE = 0.50
MIN_TRACKING_CONFIDENCE = 0.50

ACTIVE_FRAME_MARGIN_X = 0.12
ACTIVE_FRAME_MARGIN_Y = 0.12
CURSOR_SMOOTHING = 0.40
SCREEN_EDGE_PADDING = 20

PINCH_CLICK_THRESHOLD = 0.055
PINCH_RELEASE_THRESHOLD = 0.075
CLICK_COOLDOWN_SECONDS = 0.45

SHOW_DEBUG_TEXT = True
LANDMARK_RADIUS = 5
LANDMARK_COLOR = (0, 255, 255)
CONNECTION_COLOR = (80, 220, 120)
ACTIVE_FRAME_COLOR = (60, 180, 255)


HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandConnections = vision.HandLandmarksConnections.HAND_CONNECTIONS
RunningMode = vision.RunningMode


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def interpolate(value: float, start: float, end: float) -> float:
    if end <= start:
        return 0.5
    return clamp((value - start) / (end - start), 0.0, 1.0)


def normalized_distance(point_a, point_b) -> float:
    dx = point_a.x - point_b.x
    dy = point_a.y - point_b.y
    return math.sqrt(dx * dx + dy * dy)


def clamp_screen_point(x: float, y: float, screen_width: int, screen_height: int) -> tuple[float, float]:
    safe_x = clamp(x, SCREEN_EDGE_PADDING, max(screen_width - SCREEN_EDGE_PADDING, SCREEN_EDGE_PADDING))
    safe_y = clamp(y, SCREEN_EDGE_PADDING, max(screen_height - SCREEN_EDGE_PADDING, SCREEN_EDGE_PADDING))
    return safe_x, safe_y


def get_screen_size() -> tuple[int, int]:
    bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    return int(bounds.size.width), int(bounds.size.height)


def move_cursor(x: float, y: float) -> None:
    point = Quartz.CGPointMake(x, y)
    event = Quartz.CGEventCreateMouseEvent(
        None,
        Quartz.kCGEventMouseMoved,
        point,
        Quartz.kCGMouseButtonLeft,
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


def left_click(x: float, y: float) -> None:
    point = Quartz.CGPointMake(x, y)
    down_event = Quartz.CGEventCreateMouseEvent(
        None,
        Quartz.kCGEventLeftMouseDown,
        point,
        Quartz.kCGMouseButtonLeft,
    )
    up_event = Quartz.CGEventCreateMouseEvent(
        None,
        Quartz.kCGEventLeftMouseUp,
        point,
        Quartz.kCGMouseButtonLeft,
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, down_event)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, up_event)


def ensure_model_exists() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def create_landmarker() -> HandLandmarker:
    model_path = ensure_model_exists()
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    return HandLandmarker.create_from_options(options)


def draw_hand(frame, hand_landmarks) -> None:
    frame_height, frame_width = frame.shape[:2]

    for connection in HandConnections:
        start = hand_landmarks[connection.start]
        end = hand_landmarks[connection.end]
        start_point = (int(start.x * frame_width), int(start.y * frame_height))
        end_point = (int(end.x * frame_width), int(end.y * frame_height))
        cv2.line(frame, start_point, end_point, CONNECTION_COLOR, 2)

    for landmark in hand_landmarks:
        point = (int(landmark.x * frame_width), int(landmark.y * frame_height))
        cv2.circle(frame, point, LANDMARK_RADIUS, LANDMARK_COLOR, -1)


def main() -> None:
    screen_width, screen_height = get_screen_size()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, PREVIEW_WIDTH, PREVIEW_HEIGHT)

    smoothed_x = screen_width / 2
    smoothed_y = screen_height / 2
    pinch_active = False
    last_click_time = 0.0
    status_text = "Show one hand to control the cursor"
    debug_text = "No hand detected yet"

    with create_landmarker() as landmarker:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Could not read a frame from the webcam.")

                frame = cv2.flip(frame, 1)
                frame_height, frame_width = frame.shape[:2]

                left = int(frame_width * ACTIVE_FRAME_MARGIN_X)
                right = int(frame_width * (1.0 - ACTIVE_FRAME_MARGIN_X))
                top = int(frame_height * ACTIVE_FRAME_MARGIN_Y)
                bottom = int(frame_height * (1.0 - ACTIVE_FRAME_MARGIN_Y))

                cv2.rectangle(frame, (left, top), (right, bottom), ACTIVE_FRAME_COLOR, 2)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(time.monotonic() * 1000)
                results = landmarker.detect_for_video(mp_image, timestamp_ms)

                if results.hand_landmarks:
                    hand_landmarks = results.hand_landmarks[0]
                    draw_hand(frame, hand_landmarks)

                    index_tip = hand_landmarks[HandLandmark.INDEX_FINGER_TIP]
                    thumb_tip = hand_landmarks[HandLandmark.THUMB_TIP]
                    wrist = hand_landmarks[HandLandmark.WRIST]
                    middle_mcp = hand_landmarks[HandLandmark.MIDDLE_FINGER_MCP]

                    cursor_norm_x = interpolate(index_tip.x, ACTIVE_FRAME_MARGIN_X, 1.0 - ACTIVE_FRAME_MARGIN_X)
                    cursor_norm_y = interpolate(index_tip.y, ACTIVE_FRAME_MARGIN_Y, 1.0 - ACTIVE_FRAME_MARGIN_Y)

                    target_x = cursor_norm_x * screen_width
                    target_y = cursor_norm_y * screen_height

                    smoothed_x += (target_x - smoothed_x) * CURSOR_SMOOTHING
                    smoothed_y += (target_y - smoothed_y) * CURSOR_SMOOTHING
                    smoothed_x, smoothed_y = clamp_screen_point(
                        smoothed_x,
                        smoothed_y,
                        screen_width,
                        screen_height,
                    )

                    try:
                        move_cursor(smoothed_x, smoothed_y)
                    except Exception as exc:
                        status_text = "Mouse move failed - check Accessibility permissions"
                        debug_text = f"move error: {type(exc).__name__}"
                    else:
                        status_text = "Cursor control active"
                        debug_text = f"hand=yes cursor=({int(smoothed_x)}, {int(smoothed_y)})"

                    palm_scale = max(normalized_distance(wrist, middle_mcp), 0.001)
                    pinch_distance = normalized_distance(index_tip, thumb_tip) / palm_scale
                    now = time.monotonic()

                    if not pinch_active and pinch_distance <= PINCH_CLICK_THRESHOLD:
                        pinch_active = True
                        if now - last_click_time >= CLICK_COOLDOWN_SECONDS:
                            try:
                                left_click(smoothed_x, smoothed_y)
                            except Exception as exc:
                                status_text = "Click failed - check Accessibility permissions"
                                debug_text = f"click error: {type(exc).__name__}"
                            else:
                                last_click_time = now
                                status_text = "Pinch detected - click"
                                debug_text = f"pinch={pinch_distance:.2f} click=ok"
                        else:
                            status_text = "Pinch detected - cooldown"
                    elif pinch_active and pinch_distance >= PINCH_RELEASE_THRESHOLD:
                        pinch_active = False

                    index_x = int(index_tip.x * frame_width)
                    index_y = int(index_tip.y * frame_height)
                    thumb_x = int(thumb_tip.x * frame_width)
                    thumb_y = int(thumb_tip.y * frame_height)
                    cv2.circle(frame, (index_x, index_y), 8, (0, 255, 255), -1)
                    cv2.circle(frame, (thumb_x, thumb_y), 8, (255, 120, 0), -1)
                    cv2.line(frame, (index_x, index_y), (thumb_x, thumb_y), (255, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"pinch={pinch_distance:.2f}",
                        (12, 64),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    pinch_active = False
                    status_text = "Show one hand to control the cursor"
                    debug_text = "hand=no"

                cv2.putText(
                    frame,
                    status_text,
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if SHOW_DEBUG_TEXT:
                    cv2.putText(
                        frame,
                        debug_text,
                        (12, 94),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (200, 255, 200),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    frame,
                    'Index finger = cursor | Pinch = click | m = test move | c = test click | q = quit',
                    (12, frame_height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("m"):
                    try:
                        center_x, center_y = clamp_screen_point(
                            screen_width / 2,
                            screen_height / 2,
                            screen_width,
                            screen_height,
                        )
                        move_cursor(center_x, center_y)
                        status_text = "Manual move test OK"
                        debug_text = f"manual move -> ({int(center_x)}, {int(center_y)})"
                    except Exception as exc:
                        status_text = "Manual move test failed"
                        debug_text = f"manual move error: {type(exc).__name__}"
                elif key == ord("c"):
                    try:
                        center_x, center_y = clamp_screen_point(
                            screen_width / 2,
                            screen_height / 2,
                            screen_width,
                            screen_height,
                        )
                        left_click(center_x, center_y)
                        status_text = "Manual click test OK"
                        debug_text = "manual click=ok"
                    except Exception as exc:
                        status_text = "Manual click test failed"
                        debug_text = f"manual click error: {type(exc).__name__}"
                elif key == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
