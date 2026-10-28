import os
import time
import threading
import urllib.request
from pathlib import Path
from collections import deque
from datetime import datetime

import av
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import WebRtcMode, webrtc_streamer

load_dotenv()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AttentionAI",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: radial-gradient(
        circle at top left,
        #142b3b, #0b1220 55%, #090e18
    );
    color: #eef5ff;
}
.block-container {
    padding-top: 1.3rem;
    max-width: 1450px;
}
.hero {
    padding: 1.4rem;
    border: 1px solid #284154;
    border-radius: 20px;
    background: linear-gradient(120deg, #163648, #141f34);
    margin-bottom: 1rem;
}
.hero h1 {
    margin: 0;
    color: #f4f9ff;
}
.hero p {
    color: #b8c9dc;
}
div[data-testid="stMetric"] {
    background: #131f30;
    border: 1px solid #26394d;
    padding: 14px;
    border-radius: 14px;
}
</style>

<div class="hero">
    <h1>🧠 AttentionAI</h1>
    <p>
        Real-Time Attention Monitoring | Streamlit | MediaPipe Tasks
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- MODEL DOWNLOAD ----------------
MODEL_PATH = Path(__file__).parent / "face_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)


@st.cache_resource
def ensure_model():
    """Download the official face landmark model once."""

    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return str(MODEL_PATH)

    request = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "AttentionAI/1.0"}
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        model_data = response.read()

    if len(model_data) < 1_000_000:
        raise RuntimeError("The downloaded model file is invalid.")

    MODEL_PATH.write_bytes(model_data)

    return str(MODEL_PATH)


# ---------------- SHARED STATE ----------------
if "state" not in st.session_state:
    st.session_state.state = {
        "status": "Waiting for camera",
        "ear": 0.0,
        "blinks": 0,
        "closed": 0.0,
        "away": 0.0,
        "fps": 0.0,
        "score": 100,
        "frames": 0,
        "events": []
    }

if "state_lock" not in st.session_state:
    st.session_state.state_lock = threading.Lock()

STATE = st.session_state.state
STATE_LOCK = st.session_state.state_lock


# MediaPipe Face Landmarker eye landmark indexes
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ---------------- HELPER FUNCTIONS ----------------
def distance(point_a, point_b):
    return float(
        np.linalg.norm(
            np.asarray(point_a) - np.asarray(point_b)
        )
    )


def calculate_ear(landmarks, indexes, width, height):
    """Calculate Eye Aspect Ratio."""

    points = [
        (
            landmarks[index].x * width,
            landmarks[index].y * height
        )
        for index in indexes
    ]

    horizontal = distance(points[0], points[3])

    if horizontal < 1e-6:
        return 0.0

    vertical_1 = distance(points[1], points[5])
    vertical_2 = distance(points[2], points[4])

    return (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)


# ---------------- VIDEO PROCESSOR ----------------
class AttentionProcessor:

    def __init__(self, state, lock, settings):

        self.state = state
        self.lock = lock
        self.settings = settings

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=ensure_model()
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )

        self.detector = (
            vision.FaceLandmarker.create_from_options(options)
        )

        self.closed_since = None
        self.away_since = None
        self.was_closed = False

        self.blinks = 0
        self.frames = 0
        self.start_time = time.time()
        self.last_timestamp_ms = 0
        self.last_status = ""

        self.events = deque(maxlen=500)

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)

        height, width = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = max(
            int(time.time() * 1000),
            self.last_timestamp_ms + 1
        )
        self.last_timestamp_ms = timestamp

        result = self.detector.detect_for_video(
            mp_image,
            timestamp
        )

        now = time.time()
        self.frames += 1

        status = "No face detected"
        ear_value = 0.0
        closed_duration = 0.0
        away_duration = 0.0
        score = 0

        # -------- FACE DETECTED --------
        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            left_ear = calculate_ear(
                landmarks, LEFT_EYE, width, height
            )

            right_ear = calculate_ear(
                landmarks, RIGHT_EYE, width, height
            )

            ear_value = (left_ear + right_ear) / 2.0

            # Approximate face-position heuristic.
            face_left = landmarks[234].x
            face_right = landmarks[454].x
            nose_x = landmarks[1].x

            face_width = max(
                face_right - face_left, 0.001
            )

            nose_position = (
                nose_x - face_left
            ) / face_width

            looking_away = (
                nose_position < 0.34
                or nose_position > 0.66
            )

            # -------- EYE CLOSURE --------
            eyes_closed = (
                ear_value < self.settings["threshold"]
            )

            if eyes_closed:

                if self.closed_since is None:
                    self.closed_since = now

                closed_duration = now - self.closed_since

                if closed_duration > 0.08:
                    self.was_closed = True

            else:

                if self.was_closed:
                    self.blinks += 1

                self.was_closed = False
                self.closed_since = None

            # -------- LOOK-AWAY DURATION --------
            if looking_away:

                if self.away_since is None:
                    self.away_since = now

                away_duration = now - self.away_since

            else:
                self.away_since = None

            # -------- ATTENTION STATUS --------
            if closed_duration >= self.settings["drowsy"]:
                status = "DROWSY"
                score = 15

            elif away_duration >= self.settings["distracted"]:
                status = "DISTRACTED"
                score = 45

            elif eyes_closed:
                status = "Eyes closing"
                score = 65

            else:
                status = "FOCUSED"
                score = 100

            # -------- DRAW EYE LANDMARKS --------
            if self.settings["landmarks"]:

                for index in LEFT_EYE + RIGHT_EYE:

                    x = int(landmarks[index].x * width)
                    y = int(landmarks[index].y * height)

                    cv2.circle(
                        image,
                        (x, y),
                        2,
                        (64, 220, 255),
                        -1
                    )

            x1 = max(0, int(face_left * width))
            x2 = min(width - 1, int(face_right * width))

            y1 = max(0, int(landmarks[10].y * height))
            y2 = min(
                height - 1,
                int(landmarks[152].y * height)
            )

            if status == "FOCUSED":
                color = (60, 220, 120)
            elif status == "DROWSY":
                color = (40, 40, 255)
            else:
                color = (0, 190, 255)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                image,
                f"{status} | EAR {ear_value:.2f}",
                (max(8, x1), max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

            if status in ("DROWSY", "DISTRACTED"):

                message = (
                    "TAKE A BREAK"
                    if status == "DROWSY"
                    else "LOOK AT SCREEN"
                )

                cv2.putText(
                    image,
                    message,
                    (18, height - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (40, 40, 255),
                    2
                )

        # -------- EVENT LOGGING --------
        if (
            status != self.last_status
            and status in ("DROWSY", "DISTRACTED", "FOCUSED")
        ):

            if status in ("DROWSY", "DISTRACTED"):

                event = {
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Status": status,
                    "Event": f"Warning: {status.lower()} behavior detected."
                }

                self.events.append(event)

            elif self.last_status in ("DROWSY", "DISTRACTED"):

                self.events.append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Status": status,
                    "Event": "Focus returned to normal."
                })

        self.last_status = status

        with self.lock:

            self.state.update({
                "status": status,
                "ear": round(ear_value, 3),
                "blinks": self.blinks,
                "closed": round(closed_duration, 1),
                "away": round(away_duration, 1),
                "fps": round(
                    self.frames /
                    max(now - self.start_time, 0.001),
                    1
                ),
                "score": score,
                "frames": self.frames,
                "events": list(self.events)
            })

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )

    def __del__(self):
        try:
            self.detector.close()
        except Exception:
            pass


# ---------------- SIDEBAR SETTINGS ----------------
with st.sidebar:

    st.header("⚙️ Monitor Settings")

    threshold = st.slider(
        "Eye-closure sensitivity (EAR)",
        0.12, 0.32,
        float(os.getenv("EAR_THRESHOLD", "0.21")),
        0.01
    )

    drowsy = st.slider(
        "Closed-eye alert delay (seconds)",
        0.8, 4.0,
        float(os.getenv("DROWSY_SECONDS", "1.8")),
        0.2
    )

    distracted = st.slider(
        "Look-away alert delay (seconds)",
        0.8, 5.0,
        float(os.getenv("DISTRACTED_SECONDS", "2.0")),
        0.2
    )

    show_landmarks = st.checkbox(
        "Show eye landmarks",
        value=True
    )

    st.caption(
        "Educational prototype. Camera frames are not saved by this code."
    )


settings = {
    "threshold": threshold,
    "drowsy": drowsy,
    "distracted": distracted,
    "landmarks": show_landmarks
}


# # ---------------- MAIN DASHBOARD ----------------
st.subheader("📹 Live Monitoring")

left, right = st.columns([1.6, 1])

with left:
    try:
        # Downloads the model on the first run.
        ensure_model()

        webrtc_ctx = webrtc_streamer(
            key="attentionai-camera",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]}
                ]
            },
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            video_processor_factory=lambda: AttentionProcessor(
                STATE, STATE_LOCK, settings
            ),
            async_processing=True,
            desired_playing_state=True,
        )

        st.caption(
            "Press START if prompted, then allow camera access. "
            "The status label is drawn directly on the live video."
        )
        if not webrtc_ctx.state.playing:
            st.info("Camera is not running yet. Click START above and allow camera access.")
    except Exception as error:
        st.error(f"Could not initialize Face Landmarker: {error}")
        st.info(
            "Check your internet connection for the first model download, "
            "then reload the app."
        )

with right:
    # A fragment refreshes this panel independently of the whole app.
    # Without this, the video can be live while the metrics remain frozen.
    @st.fragment(run_every=1)
    def live_dashboard():
        with STATE_LOCK:
            snapshot = dict(STATE)
            event_rows = list(STATE["events"])

        status = snapshot["status"]
        status_styles = {
            "FOCUSED": ("🟢", "success"),
            "DISTRACTED": ("🟠", "warning"),
            "DROWSY": ("🔴", "error"),
            "Eyes closing": ("🟡", "warning"),
            "No face detected": ("⚪", "info"),
            "Waiting for camera": ("⚪", "info"),
        }
        emoji, message_type = status_styles.get(status, ("⚪", "info"))

        st.markdown("### Live attention status")
        st.markdown(
            f"""
            <div style="
                padding:18px 16px;
                border-radius:16px;
                margin-bottom:12px;
                background:#131f30;
                border:2px solid {'#3bd477' if status == 'FOCUSED' else '#ff5c5c' if status == 'DROWSY' else '#ffb347' if status in ('DISTRACTED', 'Eyes closing') else '#526579'};
                text-align:center;">
                <div style="font-size:34px;font-weight:800;">{emoji} {status}</div>
                <div style="font-size:13px;color:#b8c9dc;margin-top:5px;">Updates while the camera is running</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        col1.metric("Attention score", f'{snapshot["score"]}%')
        col2.metric("Blink count", snapshot["blinks"])

        col3, col4 = st.columns(2)
        col3.metric("Eye aspect ratio", f'{snapshot["ear"]:.3f}')
        col4.metric("Processing FPS", snapshot["fps"])

        st.progress(max(0, min(100, int(snapshot["score"]))) / 100)
        st.caption(
            f'Closed eyes: {snapshot["closed"]} s | '
            f'Look-away: {snapshot["away"]} s'
        )

        if status == "DROWSY":
            st.error("⚠️ Eyes appear closed for an extended period. Consider resting.")
        elif status == "DISTRACTED":
            st.warning("👀 Your face appears turned away from the camera.")
        elif status == "FOCUSED":
            st.success("You appear focused based on the current camera signals.")
        elif status == "Eyes closing":
            st.warning("Eye closure detected — monitoring duration.")
        else:
            st.info("Make sure your face is visible and well lit.")

        st.divider()
        st.subheader("📊 Session Analytics")
        a, b, c = st.columns(3)
        a.metric("Frames processed", snapshot["frames"])
        b.metric("Current state", status)
        c.metric("Logged events", len(event_rows))

        if event_rows:
            df = pd.DataFrame(event_rows)
            st.dataframe(
                df.tail(12),
                use_container_width=True,
                hide_index=True
            )
            st.download_button(
                "⬇️ Download session log (CSV)",
                df.to_csv(index=False).encode("utf-8"),
                file_name="attentionai_session_log.csv",
                mime="text/csv",
                key="download-session-log"
            )

    live_dashboard()

with st.expander("ℹ️ How it works and limitations"):
    st.markdown("""
    - **Live video overlay:** status and eye ratio are drawn on each processed webcam frame.
    - **EAR:** estimates eye openness from facial landmarks.
    - **Drowsiness:** triggers when eye closure remains below the threshold.
    - **Look-away:** uses a rough face-position heuristic, not calibrated gaze tracking.
    - **Attention score:** illustrative and not scientifically validated.
    - Lighting, glasses, camera position, and face angle can affect results.
    - This is an educational prototype, not a medical device or certified safety system.
    """)

st.markdown("---")
st.caption("AttentionAI | MediaPipe Tasks API")
