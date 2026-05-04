# test_tcn_adavu_video.py
import cv2
import numpy as np
import tensorflow as tf
import pickle
from collections import Counter
import mediapipe as mp

# -------------------------
# 1. LOAD MODEL & LABEL MAP
# -------------------------
MODEL_PATH = "best_tcn_adavu_model.h5"   # path to your TCN model
LABEL_PATH = "label_map (1).pkl"             # path to label map

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_PATH, "rb") as f:
    label_map = pickle.load(f)
id_to_label = {v: k for k, v in label_map.items()}

# -------------------------
# 2. CONFIG
# -------------------------
WINDOW_SIZE = 60          # same as used in training
STEP_SIZE = 10
OUTPUT_VIDEO = "21.mp4"

# -------------------------
# 3. MEDIAPIPE POSE
# -------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# Use all 33 keypoints (x, y, z, visibility) = 33*4 = 132 features
KEYPOINT_IDS = list(range(33))

def extract_keypoints(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(frame_rgb)
    if not res.pose_landmarks:
        return None
    landmarks = res.pose_landmarks.landmark
    pts = []
    for i in KEYPOINT_IDS:
        pts.extend([
            landmarks[i].x,
            landmarks[i].y,
            landmarks[i].z,
            landmarks[i].visibility
        ])
    return np.array(pts, dtype=np.float32)

# -------------------------
# 4. SLIDING WINDOW
# -------------------------
def create_windows(seq, window_size=WINDOW_SIZE, step=STEP_SIZE):
    windows = []
    for i in range(0, len(seq) - window_size + 1, step):
        windows.append(seq[i:i+window_size])
    return windows

def normalize_relative(X):
    X = X.copy()
    for i in range(X.shape[0]):
        center = X[i,:,0:2].mean(axis=1, keepdims=True)
        X[i,:,0:2] -= center
    return X

# -------------------------
# 5. READ VIDEO
# -------------------------
VIDEO_PATH = "video7.mp4"  # your input video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

sequence = []
predictions = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    kp = extract_keypoints(frame)
    if kp is None:
        continue

    sequence.append(kp)

    # Sliding window prediction
    if len(sequence) >= WINDOW_SIZE:
        window_seq = np.array(sequence[-WINDOW_SIZE:])
        window_seq = np.expand_dims(window_seq, axis=0)
        window_seq = normalize_relative(window_seq)
        
        pred = model.predict(window_seq, verbose=0)
        cls = np.argmax(pred)
        confidence = np.max(pred)
        predictions.append(cls)

        # Overlay predicted label
        
        cv2.putText(
            frame,
            "Classified Adavu:",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"{id_to_label[cls]} ({confidence*100:.1f}%)",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )

    out.write(frame)

cap.release()
out.release()
pose.close()

# -------------------------
# 6. FINAL VIDEO PREDICTION SUMMARY
# -------------------------
if predictions:
    final_class = Counter(predictions).most_common(1)[0][0]
    print("Predicted Adavu for the video:", id_to_label[final_class])
else:
    print("No keypoints detected. Cannot predict.")

print("Saved prediction video as:", OUTPUT_VIDEO)
