"""
Unified auto-detect utility for macOS.

Features:
    - Motion detection via webcam
    - Clap detection via microphone
    - Launches apps when motion is followed by a clap
    - Hand tracking for mouse control
    - Left click when the index finger bends
    - Right click when index + middle fingertips touch

Install dependencies:
    brew install portaudio
    pip install numpy opencv-python pyaudio mediapipe pyobjc-framework-Quartz

On first run, the script downloads the MediaPipe hand landmarker model into
the local project folder.

On macOS, allow camera, microphone, and Accessibility control for Python/your IDE.
Press "q" in the preview window to quit.
"""

from dataclasses import dataclass, field
from pathlib import Path
import math
import subprocess
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import pyaudio
import Quartz
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark


# =========================
# Tuning constants
# =========================
CAMERA_INDEX = 0
WINDOW_NAME = "Auto Detect Control"
PREVIEW_WIDTH = 1100
PREVIEW_HEIGHT = 760

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

APP_NAMES = [
    "Codex",
    "Google Chrome",
    "Visual Studio Code",
    "Telegram",
]

MOTION_DIFF_THRESHOLD = 25
MOTION_MIN_CONTOUR_AREA = 2500
MOTION_BLUR_KERNEL = (21, 21)
MOTION_DILATE_ITERATIONS = 2

CLAP_LISTEN_TIMEOUT = 5.0
AUDIO_RATE = 44100
AUDIO_CHUNK = 1024
AUDIO_CHANNELS = 1
CLAP_RMS_THRESHOLD = 1800.0
CLAP_RESET_RATIO = 0.55
CLAP_MAX_HIGH_CHUNKS = 3
LAUNCH_STATUS_SECONDS = 2.0
LAUNCH_COOLDOWN_SECONDS = 3.0

NUM_HANDS = 1
MIN_HAND_DETECTION_CONFIDENCE = 0.50
MIN_HAND_PRESENCE_CONFIDENCE = 0.50
MIN_TRACKING_CONFIDENCE = 0.50

ACTIVE_FRAME_MARGIN_X = 0.12
ACTIVE_FRAME_MARGIN_Y = 0.12
CURSOR_SMOOTHING = 0.40
SCREEN_EDGE_PADDING = 20

INDEX_BENT_ANGLE_THRESHOLD = 145.0
RIGHT_TOUCH_THRESHOLD = 0.26
CLICK_COOLDOWN_SECONDS = 0.45

SHOW_DEBUG_TEXT = True
LANDMARK_RADIUS = 5
LANDMARK_COLOR = (0, 255, 255)
CONNECTION_COLOR = (80, 220, 120)
ACTIVE_FRAME_COLOR = (60, 180, 255)
MOTION_BOX_COLOR = (0, 255, 0)


HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandConnections = vision.HandLandmarksConnections.HAND_CONNECTIONS
RunningMode = vision.RunningMode


@dataclass
class DetectionState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    waiting_for_clap: bool = False
    clap_deadline: float = 0.0
    launch_status_until: float = 0.0
    launch_cooldown_until: float = 0.0

    def _expire_locked(self, now: float) -> None:
        if self.waiting_for_clap and now > self.clap_deadline:
            self.waiting_for_clap = False

    def get_motion_status_text(self) -> str:
        now = time.monotonic()
        with self.lock:
            self._expire_locked(now)
            if now < self.launch_status_until:
                return "Motion/clap: launching apps!"
            if self.waiting_for_clap:
                return "Motion/clap: motion detected - waiting for clap..."
            return "Motion/clap: idle"

    def notify_motion(self) -> None:
        now = time.monotonic()
        with self.lock:
            self._expire_locked(now)
            if now < self.launch_status_until or now < self.launch_cooldown_until:
                return
            if not self.waiting_for_clap:
                self.waiting_for_clap = True
                self.clap_deadline = now + CLAP_LISTEN_TIMEOUT

    def consume_clap(self) -> bool:
        now = time.monotonic()
        with self.lock:
            self._expire_locked(now)
            if not self.waiting_for_clap or now > self.clap_deadline:
                return False

            self.waiting_for_clap = False
            self.launch_status_until = now + LAUNCH_STATUS_SECONDS
            self.launch_cooldown_until = now + LAUNCH_COOLDOWN_SECONDS
            return True


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


def angle_degrees(point_a, point_b, point_c) -> float:
    vector_ab = np.array([point_a.x - point_b.x, point_a.y - point_b.y], dtype=np.float32)
    vector_cb = np.array([point_c.x - point_b.x, point_c.y - point_b.y], dtype=np.float32)

    norm_ab = float(np.linalg.norm(vector_ab))
    norm_cb = float(np.linalg.norm(vector_cb))
    if norm_ab == 0.0 or norm_cb == 0.0:
        return 180.0

    cosine = float(np.dot(vector_ab, vector_cb) / (norm_ab * norm_cb))
    cosine = clamp(cosine, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


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


def right_click(x: float, y: float) -> None:
    point = Quartz.CGPointMake(x, y)
    down_event = Quartz.CGEventCreateMouseEvent(
        None,
        Quartz.kCGEventRightMouseDown,
        point,
        Quartz.kCGMouseButtonRight,
    )
    up_event = Quartz.CGEventCreateMouseEvent(
        None,
        Quartz.kCGEventRightMouseUp,
        point,
        Quartz.kCGMouseButtonRight,
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, down_event)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, up_event)


def launch_apps() -> None:
    for app_name in APP_NAMES:
        # macOS launch command.
        # For Windows, replace this with os.startfile(...) or subprocess.Popen([...]).
        # For Linux, replace this with xdg-open or the app executable path.
        subprocess.Popen(
            ["open", "-a", app_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


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


def detect_motion(frame, previous_gray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, MOTION_BLUR_KERNEL, 0)

    motion_detected = False
    motion_boxes = []

    if previous_gray is not None:
        frame_delta = cv2.absdiff(previous_gray, gray)
        thresh = cv2.threshold(
            frame_delta,
            MOTION_DIFF_THRESHOLD,
            255,
            cv2.THRESH_BINARY,
        )[1]
        thresh = cv2.dilate(thresh, None, iterations=MOTION_DILATE_ITERATIONS)
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            if cv2.contourArea(contour) < MOTION_MIN_CONTOUR_AREA:
                continue
            motion_detected = True
            x, y, w, h = cv2.boundingRect(contour)
            motion_boxes.append((x, y, w, h))

    return gray, motion_detected, motion_boxes


def audio_listener(state: DetectionState, stop_event: threading.Event) -> None:
    audio = pyaudio.PyAudio()
    stream = None

    high_run_chunks = 0
    peak_rms = 0.0

    try:
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK,
        )

        while not stop_event.is_set():
            data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(samples * samples)))

            if rms >= CLAP_RMS_THRESHOLD or (
                high_run_chunks > 0 and rms >= CLAP_RMS_THRESHOLD * CLAP_RESET_RATIO
            ):
                high_run_chunks += 1
                peak_rms = max(peak_rms, rms)
                continue

            if 0 < high_run_chunks <= CLAP_MAX_HIGH_CHUNKS and peak_rms >= CLAP_RMS_THRESHOLD:
                if state.consume_clap():
                    launch_apps()

            high_run_chunks = 0
            peak_rms = 0.0

    except Exception as exc:
        print(f"Audio listener stopped: {exc}")
        stop_event.set()
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        audio.terminate()


def video_loop(state: DetectionState, stop_event: threading.Event) -> None:
    screen_width, screen_height = get_screen_size()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, PREVIEW_WIDTH, PREVIEW_HEIGHT)

    previous_gray = None
    smoothed_x = screen_width / 2
    smoothed_y = screen_height / 2
    left_gesture_active = False
    right_gesture_active = False
    last_left_click_time = 0.0
    last_right_click_time = 0.0

    with create_landmarker() as landmarker:
        try:
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Could not read a frame from the webcam.")

                frame = cv2.flip(frame, 1)
                frame_height, frame_width = frame.shape[:2]

                gray, motion_detected, motion_boxes = detect_motion(frame, previous_gray)
                previous_gray = gray

                if motion_detected:
                    state.notify_motion()

                for x, y, w, h in motion_boxes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), MOTION_BOX_COLOR, 2)

                left = int(frame_width * ACTIVE_FRAME_MARGIN_X)
                right = int(frame_width * (1.0 - ACTIVE_FRAME_MARGIN_X))
                top = int(frame_height * ACTIVE_FRAME_MARGIN_Y)
                bottom = int(frame_height * (1.0 - ACTIVE_FRAME_MARGIN_Y))
                cv2.rectangle(frame, (left, top), (right, bottom), ACTIVE_FRAME_COLOR, 2)

                hand_status_text = "Hand: show one hand to control the cursor"
                debug_text = "hand=no"

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(time.monotonic() * 1000)
                results = landmarker.detect_for_video(mp_image, timestamp_ms)

                if results.hand_landmarks:
                    hand_landmarks = results.hand_landmarks[0]
                    draw_hand(frame, hand_landmarks)

                    index_tip = hand_landmarks[HandLandmark.INDEX_FINGER_TIP]
                    index_mcp = hand_landmarks[HandLandmark.INDEX_FINGER_MCP]
                    index_pip = hand_landmarks[HandLandmark.INDEX_FINGER_PIP]
                    middle_tip = hand_landmarks[HandLandmark.MIDDLE_FINGER_TIP]
                    middle_mcp = hand_landmarks[HandLandmark.MIDDLE_FINGER_MCP]
                    wrist = hand_landmarks[HandLandmark.WRIST]

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
                        hand_status_text = "Hand: mouse move failed - check Accessibility permissions"
                        debug_text = f"move error: {type(exc).__name__}"
                    else:
                        hand_status_text = "Hand: cursor control active"

                    palm_scale = max(normalized_distance(wrist, middle_mcp), 0.001)
                    right_touch_distance = normalized_distance(index_tip, middle_tip) / palm_scale
                    index_angle = angle_degrees(index_mcp, index_pip, index_tip)
                    index_bent = index_angle < INDEX_BENT_ANGLE_THRESHOLD
                    right_touch = right_touch_distance <= RIGHT_TOUCH_THRESHOLD
                    now = time.monotonic()

                    if right_touch:
                        if not right_gesture_active and now - last_right_click_time >= CLICK_COOLDOWN_SECONDS:
                            try:
                                right_click(smoothed_x, smoothed_y)
                            except Exception as exc:
                                hand_status_text = "Hand: right click failed - check Accessibility permissions"
                                debug_text = f"right click error: {type(exc).__name__}"
                            else:
                                hand_status_text = "Hand: right click"
                                last_right_click_time = now
                        right_gesture_active = True
                        left_gesture_active = False
                    else:
                        right_gesture_active = False

                        if index_bent:
                            if not left_gesture_active and now - last_left_click_time >= CLICK_COOLDOWN_SECONDS:
                                try:
                                    left_click(smoothed_x, smoothed_y)
                                except Exception as exc:
                                    hand_status_text = "Hand: left click failed - check Accessibility permissions"
                                    debug_text = f"left click error: {type(exc).__name__}"
                                else:
                                    hand_status_text = "Hand: left click"
                                    last_left_click_time = now
                            left_gesture_active = True
                        else:
                            left_gesture_active = False

                    debug_text = (
                        f"hand=yes cursor=({int(smoothed_x)}, {int(smoothed_y)}) "
                        f"index_angle={index_angle:.0f} right_touch={right_touch_distance:.2f}"
                    )

                    index_x = int(index_tip.x * frame_width)
                    index_y = int(index_tip.y * frame_height)
                    middle_x = int(middle_tip.x * frame_width)
                    middle_y = int(middle_tip.y * frame_height)
                    cv2.circle(frame, (index_x, index_y), 8, (0, 255, 255), -1)
                    cv2.circle(frame, (middle_x, middle_y), 8, (255, 120, 0), -1)
                    cv2.line(frame, (index_x, index_y), (middle_x, middle_y), (255, 255, 0), 2)
                else:
                    left_gesture_active = False
                    right_gesture_active = False

                cv2.putText(
                    frame,
                    state.get_motion_status_text(),
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.78,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    hand_status_text,
                    (12, 64),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if SHOW_DEBUG_TEXT:
                    cv2.putText(
                        frame,
                        debug_text,
                        (12, 96),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (200, 255, 200),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    frame,
                    'Move index = cursor | bend index = left click | index+middle touch = right click',
                    (12, frame_height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    'm = move center | l = left click | r = right click | q = quit',
                    (12, frame_height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("m"):
                    center_x, center_y = clamp_screen_point(
                        screen_width / 2,
                        screen_height / 2,
                        screen_width,
                        screen_height,
                    )
                    move_cursor(center_x, center_y)
                elif key == ord("l"):
                    left_click(smoothed_x, smoothed_y)
                elif key == ord("r"):
                    right_click(smoothed_x, smoothed_y)
                elif key == ord("q"):
                    stop_event.set()
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main() -> None:
    state = DetectionState()
    stop_event = threading.Event()

    audio_thread = threading.Thread(
        target=audio_listener,
        args=(state, stop_event),
        name="audio-listener",
        daemon=True,
    )
    audio_thread.start()

    try:
        video_loop(state, stop_event)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        audio_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
