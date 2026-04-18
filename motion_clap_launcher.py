"""
Motion + clap launcher for macOS.

Install dependencies:
    pip install opencv-python pyaudio

Press "q" in the preview window to quit.
"""

import subprocess
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import pyaudio


# =========================
# Tuning constants
# =========================
CAMERA_INDEX = 0
WINDOW_NAME = "Motion + Clap Launcher"
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360

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

APP_NAMES = [
    "Codex",
    "Google Chrome",
    "Visual Studio Code",
    "Telegram",
]


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

    def get_status_text(self) -> str:
        now = time.monotonic()
        with self.lock:
            self._expire_locked(now)
            if now < self.launch_status_until:
                return "Launching apps!"
            if self.waiting_for_clap:
                return "Motion detected - waiting for clap..."
            return "Idle"

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
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, PREVIEW_WIDTH, PREVIEW_HEIGHT)

    previous_gray = None

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")

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

            if motion_detected:
                state.notify_motion()

            for x, y, w, h in motion_boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            status_text = state.get_status_text()
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
            cv2.putText(
                frame,
                'Press "q" to quit',
                (12, frame.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, frame)
            previous_gray = gray

            if cv2.waitKey(1) & 0xFF == ord("q"):
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
