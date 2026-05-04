import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
from collections import deque, Counter

# ===================== CONFIG =====================
MODEL_PATH = "best_model.h5"
LABEL_MAP_PATH = "label_map.pkl"
VIDEO_PATH = "video3.mp4"
OUTPUT_PATH = "9.mp4"

WINDOW_SIZE = 60          # must match training
STRIDE = 5               # sliding step
SMOOTHING_WINDOW = 15    # temporal voting
CONF_THRESH = 0.6
FEATURES = 132
# =================================================

# Load model + labels
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, "rb") as f:
    label_map = pickle.load(f)

id_to_label = {v: k for k, v in label_map.items()}

# Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# Select same landmarks used in training
KEYPOINT_IDS = list(range(33))  # adjust ONLY if training differs

def extract_keypoints(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img)

    if not res.pose_landmarks:
        return None

    lm = res.pose_landmarks.landmark
    pts = []
    for i in KEYPOINT_IDS:
        pts.extend([lm[i].x, lm[i].y, lm[i].z, lm[i].visibility])

    pts = np.array(pts)

    if pts.shape[0] != FEATURES:
        return None

    return pts


# Video IO
cap = cv2.VideoCapture(VIDEO_PATH)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

sequence = deque(maxlen=WINDOW_SIZE)
prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)

current_label = "Waiting..."

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    kp = extract_keypoints(frame)
    if kp is None:
        out.write(frame)
        continue

    sequence.append(kp)

    if len(sequence) == WINDOW_SIZE and frame_idx % STRIDE == 0:
        window = np.expand_dims(np.array(sequence), axis=0)

        preds = model.predict(window, verbose=0)[0]
        cls = np.argmax(preds)
        conf = preds[cls]

        if conf > CONF_THRESH:
            prediction_buffer.append(cls)

        if len(prediction_buffer) > 5:
            majority = Counter(prediction_buffer).most_common(1)[0][0]
            current_label = id_to_label[majority]

    # Draw label
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(
        frame,
        current_label,
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2
    )

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

print("✅ Output saved to:", OUTPUT_PATH)
