# AttentionAI — Real-Time Attention & Drowsiness Monitor

A Streamlit computer-vision prototype inspired by real-time attention-monitoring dashboards. It uses a webcam, OpenCV, MediaPipe Face Mesh and Streamlit WebRTC to estimate eye closure, count blinks, flag prolonged eye closure, and show a rough look-away warning.

## Features

- Live webcam stream inside Streamlit using WebRTC
- MediaPipe facial landmark detection
- Eye Aspect Ratio (EAR) calculation for both eyes
- Blink counter and prolonged eye-closure warning
- Approximate look-away warning using face position (not calibrated eye tracking)
- Live status, attention indicator, EAR and FPS metrics
- Recent event log with CSV download
- Adjustable thresholds and landmark overlay
- Local frame processing; no cloud AI API key is required

## Project files

```text
AttentionAI_Streamlit/
├── app.py
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
└── README.md
```

## Requirements

**Recommended: Python 3.11 (64-bit).** MediaPipe wheels may not be available for every Python version/platform; Python 3.11 is the safest choice for this dependency set. Python 3.14 is not recommended for this project.

You need:
- Python 3.11
- A webcam
- VS Code or another terminal
- A modern browser with camera permissions

## Run on Windows (VS Code)

Open the project folder in VS Code, then open **Terminal → New Terminal**.

### 1. Create a virtual environment

```powershell
py -3.11 -m venv .venv
```

### 2. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this in the current terminal and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Upgrade pip and install dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Start the app

```powershell
python -m streamlit run app.py
```

Open the local URL printed in the terminal, normally `http://localhost:8501`. Allow camera access in the browser.

## Run on macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -m streamlit run app.py
```

## How to use

1. Start the app and allow browser camera access.
2. Sit in good lighting with your face visible.
3. Start the webcam stream in the Live monitoring section.
4. Watch the current state, EAR, blink count and attention indicator.
5. Adjust the EAR threshold and warning delays in the sidebar if needed.
6. Download the CSV event log when events have been recorded.

## How it works

1. WebRTC sends camera frames to the Python video processor.
2. MediaPipe estimates face landmarks.
3. The app calculates the Eye Aspect Ratio (EAR) from six landmarks around each eye.
4. Sustained low EAR triggers a drowsiness warning.
5. A simple face-position heuristic can flag possible look-away behavior.
6. State changes are logged and can be downloaded as CSV.

## Troubleshooting

### `No matching distribution found for mediapipe`
Use Python 3.11 (64-bit), recreate the virtual environment, then reinstall requirements.

### Camera is blank or permission is denied
- Allow camera access in browser settings.
- Close apps such as Zoom, Teams or Windows Camera that may be using the webcam.
- Refresh the page and start the stream again.

### WebRTC does not connect
- Try the app locally first.
- On hosted deployments, use HTTPS and ensure the host/network permits WebRTC traffic.
- Some corporate networks, VPNs or firewalls block peer connections.

### Installation error with NumPy
This project pins NumPy below version 2 for compatibility with the selected MediaPipe release. Reinstall with `pip install -r requirements.txt` in the activated virtual environment.

## Privacy

The video is processed in the running Python app and is not written to disk by this code. The CSV download contains event timestamps/statuses, not video. If you deploy this app on a server, camera frames are sent from the browser to that server for processing; disclose this to users and configure your hosting environment accordingly.

## Important limitations

This is an educational prototype, not a reproduction of the original LinkedIn author's private source code. EAR-based eye closure is a heuristic. Look-away detection is only a rough face-position proxy; it is not true gaze estimation. The attention score is a simple illustrative rule, not a validated measurement. Lighting, glasses, head pose and camera placement affect accuracy. **Do not use this prototype as a driver-safety system or for medical/disciplinary decisions.**

## Tech stack

Python · Streamlit · Streamlit WebRTC · OpenCV · MediaPipe · NumPy · Pandas
