import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
# ===================== CONFIG =====================

VIDEO_PATH = "/Users/harshaprada/ProjectX/video.mp4"
MODEL_PATH = "/Users/harshaprada/ProjectX/best_model.h5"
LABEL_MAP_PATH = "/Users/harshaprada/ProjectX/label_map.pkl"
OUTPUT_PATH = "/Users/harshaprada/ProjectX/output_annotated2.mp4"

MAX_FRAMES = 50
FEATURES = 132
STRIDE = 1         # move window by 1 frame for smooth live prediction
CONF_THRESHOLD = 0.6  # only show prediction if confidence >= 0.6

# ===================== LOAD MODEL =====================
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_MAP_PATH, "rb") as f:
    label_map = pickle.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

# ===================== MEDIAPIPE SETUP =====================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===================== HELPERS =====================
def extract_keypoints(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if not results.pose_landmarks:
        return None
    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(keypoints)

def normalize_relative(seq):
    seq = seq.copy()
    center = seq[:, 0:2].mean(axis=1, keepdims=True)
    seq[:, 0:2] -= center
    return seq

def fix_sequence_length(seq, max_frames=MAX_FRAMES):
    T = seq.shape[0]
    if T > max_frames:
        return seq[:max_frames]
    if T < max_frames:
        pad_len = max_frames - T
        pad = np.repeat(seq[-1][None, :], pad_len, axis=0)
        return np.vstack([seq, pad])
    return seq

# ===================== VIDEO PROCESSING =====================
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Setup video writer
out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

sequence = []
pred_label = ""
pred_conf = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    keypoints = extract_keypoints(frame)
    if keypoints is not None:
        sequence.append(keypoints)
    
    # Only start prediction when we have enough frames
    if len(sequence) >= MAX_FRAMES:
        window = np.array(sequence[-MAX_FRAMES:])
        window = normalize_relative(window)
        X = np.expand_dims(window, axis=0)
        pred = model.predict(X, verbose=0)
        pred_class = int(np.argmax(pred))
        pred_conf = float(np.max(pred))
        if pred_conf >= CONF_THRESHOLD:
            pred_label = inv_label_map[pred_class]
        else:
            pred_label = "uncertain"

    # Draw label + confidence on frame
    cv2.putText(
        frame,
        f"{pred_label} ({round(pred_conf*100,2)}%)",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    # Show the frame
    cv2.imshow("Bharatanatyam Prediction", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
pose.close()
print("Done! Video saved to:", OUTPUT_PATH)