import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
from collections import deque, Counter

# =========================
# CONFIG
# =========================
MODEL_PATH = "best_tcn_adavu_model.h5"
LABEL_MAP_PATH = "label_map (1).pkl"
INPUT_VIDEO = "video8.mp4"
OUTPUT_VIDEO = "20.mp4"

WINDOW_SIZE = 60
FEATURES = 132
CONF_THRESH = 0.6
SMOOTHING_WINDOW = 15
STABILITY_COUNT = 8

# =========================
# LOAD MODEL & LABELS
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, "rb") as f:
    label_map = pickle.load(f)

id_to_label = {v: k for k, v in label_map.items()}

# =========================
# MEDIAPIPE SETUP
# =========================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# KEYPOINT EXTRACTION
# =========================
def extract_keypoints(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if not res.pose_landmarks:
        return None

    pts = []
    for lm in res.pose_landmarks.landmark:
        pts.extend([lm.x, lm.y, lm.z, lm.visibility])

    return np.array(pts, dtype=np.float32)

# =========================
# VIDEO IO
# =========================
cap = cv2.VideoCapture(INPUT_VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

# =========================
# TEMPORAL SMOOTHING
# =========================
sequence = deque(maxlen=WINDOW_SIZE)
pred_buffer = deque(maxlen=SMOOTHING_WINDOW)

stable_class = None
stable_text = "Identifying Adavu..."

# =========================
# PROCESS VIDEO
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    kp = extract_keypoints(frame)
    if kp is None:
        out.write(frame)
        continue

    sequence.append(kp)

    if len(sequence) == WINDOW_SIZE:
        window = np.expand_dims(np.array(sequence), axis=0)
        pred = model.predict(window, verbose=0)[0]

        cls = np.argmax(pred)
        conf = pred[cls]

        if conf > CONF_THRESH:
            pred_buffer.append(cls)

        if len(pred_buffer) == SMOOTHING_WINDOW:
            common, count = Counter(pred_buffer).most_common(1)[0]

            if count >= STABILITY_COUNT:
                stable_class = common
                name = id_to_label[stable_class]
                name = name.replace("adavu", " Adavu").title()
                stable_text = f"Recognized Adavu: {name}"

    # =========================
    # ELEGANT OVERLAY
    # =========================
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    cv2.putText(
        frame,
        stable_text,
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 0),
        3,
        cv2.LINE_AA
    )

    out.write(frame)

# =========================
# CLEANUP
# =========================
cap.release()
out.release()
pose.close()

print("✅ Output saved as:", OUTPUT_VIDEO)
