import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
from collections import deque, Counter

# ================= CONFIG =================
MODEL_PATH = "best_model.h5"
LABEL_MAP_PATH = "label_map.pkl"
VIDEO_PATH = "video3.mp4"
OUTPUT_PATH = "6.mp4"

WINDOW_SIZE = 60          # SAME as training
FEATURES = 132
VOTE_WINDOW = 15          # ~2 sec
LOCK_FRAMES = 45          # ~1.5 sec
CONF_THRESHOLD = 0.6
# ==========================================

# Load model & labels
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_MAP_PATH, "rb") as f:
    label_map = pickle.load(f)

inv_label_map = {v: k for k, v in label_map.items()}

# MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------- Keypoint extraction (132) --------
def extract_keypoints(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img)

    if not res.pose_landmarks:
        return np.zeros(FEATURES)

    lm = res.pose_landmarks.landmark
    keypoints = []

    for p in lm:
        keypoints.extend([p.x, p.y, p.z, p.visibility])

    return np.array(keypoints, dtype=np.float32)

# -------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)
w, h = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

sequence = deque(maxlen=WINDOW_SIZE)
pred_buffer = deque(maxlen=VOTE_WINDOW)

current_label = "Waiting"
lock_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    kp = extract_keypoints(frame)
    sequence.append(kp)

    if len(sequence) == WINDOW_SIZE and lock_count == 0:
        window = np.expand_dims(sequence, axis=0)
        preds = model.predict(window, verbose=0)[0]
        pred_label = np.argmax(preds)
        pred_buffer.append(pred_label)

        if len(pred_buffer) == VOTE_WINDOW:
            vote = Counter(pred_buffer)
            label, count = vote.most_common(1)[0]

            if count / VOTE_WINDOW >= CONF_THRESHOLD:
                current_label = inv_label_map[label]
                lock_count = LOCK_FRAMES
                pred_buffer.clear()

    if lock_count > 0:
        lock_count -= 1

    # ---------- Overlay ----------
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Adavu: {current_label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    out.write(frame)

cap.release()
out.release()
pose.close()

print("✅ Output saved as:", OUTPUT_PATH)
