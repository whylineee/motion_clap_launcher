# Auto Detect

Computer vision utilities for macOS built with Python, OpenCV, PyAudio, and MediaPipe.

This repository currently contains two standalone scripts:

- `motion_clap_launcher.py`: detects motion from the webcam, waits for a clap, then launches apps
- `hand_mouse_control.py`: detects a hand from the webcam and moves the mouse cursor with finger tracking

## Features

### Motion + Clap Launcher

- webcam-based motion detection using frame differencing
- clap detection from microphone input using RMS spike analysis
- launches `Codex`, `Google Chrome`, `Visual Studio Code`, and `Telegram`
- small OpenCV preview window with live status text
- tunable thresholds at the top of the file
- audio and video run in parallel using threads

### Hand Mouse Control

- hand landmark tracking with MediaPipe
- cursor control using the index finger
- pinch gesture for click
- live preview window with debug overlay
- built-in manual test keys:
  - `m` for manual cursor move test
  - `c` for manual click test
  - `q` to quit

## Requirements

- macOS
- Python 3
- webcam
- microphone

## Local macOS dependency

`PyAudio` needs `portaudio` installed on the system:

```bash
brew install portaudio
```

## Python setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install Python dependencies:

```bash
pip install numpy opencv-python pyaudio mediapipe pyobjc-framework-Quartz
```

## Run

### 1. Motion + clap launcher

```bash
python motion_clap_launcher.py
```

Behavior:

1. The app watches the webcam feed for motion.
2. When motion is detected, it starts listening for a clap for 5 seconds.
3. If a clap is detected in that time window, it launches the configured apps.

### 2. Hand mouse controller

```bash
python hand_mouse_control.py
```

Behavior:

1. Show one hand to the webcam.
2. Move your index finger to control the cursor.
3. Pinch thumb + index finger to click.

## macOS permissions

Both scripts may need permission from macOS before they work correctly.

Open:

- `System Settings -> Privacy & Security -> Camera`
- `System Settings -> Privacy & Security -> Microphone`
- `System Settings -> Privacy & Security -> Accessibility`

Grant access to the app that launches Python.

Examples:

- `Terminal`
- `iTerm`
- `PyCharm`
- `Visual Studio Code`

If you run the script from an IDE, that IDE often needs the permission, not only `python`.

## Tuning

Both scripts keep all main sensitivity values at the top of each file so they are easy to adjust.

Useful values to tune:

- `MOTION_DIFF_THRESHOLD`
- `MOTION_MIN_CONTOUR_AREA`
- `CLAP_RMS_THRESHOLD`
- `CLAP_LISTEN_TIMEOUT`
- `ACTIVE_FRAME_MARGIN_X`
- `ACTIVE_FRAME_MARGIN_Y`
- `CURSOR_SMOOTHING`
- `PINCH_CLICK_THRESHOLD`

## Project structure

```text
.
├── hand_mouse_control.py
├── motion_clap_launcher.py
└── README.md
```

## Troubleshooting

### `ModuleNotFoundError`

Make sure you are running the script from the same virtual environment where the packages were installed:

```bash
.venv/bin/python motion_clap_launcher.py
.venv/bin/python hand_mouse_control.py
```

### Webcam window opens, but apps do not launch

- check microphone permission
- try lowering `CLAP_RMS_THRESHOLD`
- reduce room noise
- make sure the clap is short and sharp

### Hand landmarks are visible, but the cursor does not move

- check `Accessibility` permission
- press `m` inside the hand control window
- if `m` does not move the cursor, the issue is system permissions rather than hand detection

### Hand is not detected reliably

- improve lighting
- keep one hand clearly visible
- move the hand closer to the camera
- lower `MIN_DETECTION_CONFIDENCE` and `MIN_TRACKING_CONFIDENCE`

## Notes

- `motion_clap_launcher.py` uses `open -a` and is written for macOS app launching
- app names can be changed in the `APP_NAMES` list
- for Windows or Linux, the launch logic would need to be replaced
