import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque, Counter

# =========================
# PATHS
# =========================
MODEL_PATH = "best_adavu_model.h5"
LABEL_PATH = "label_map.npy"
VIDEO_PATH = "Bharatanatyam_test_video.mp4"
OUTPUT_PATH = "adavu_prediction_output.mp4"

# =========================
# LOAD MODEL + LABELS
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

label_map = np.load(LABEL_PATH, allow_pickle=True).item()
id_to_label = {v: k for k, v in label_map.items()}

print("Loaded labels:", id_to_label)

# =========================
# MEDIAPIPE POSE
# =========================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True
)

# =========================
# KEYPOINT CONFIG
# (MUST MATCH TRAINING)
# =========================
KEYPOINT_IDS = [
    23, 24,   # hips
    25, 26,   # knees
    27, 28    # ankles
]

NUM_KEYPOINTS = len(KEYPOINT_IDS) * 2
WINDOW_SIZE = 60
CONF_THRESHOLD = 0.6

# =========================
# TEMPORAL SMOOTHING
# =========================
PRED_BUFFER = deque(maxlen=15)
DISPLAY_LABEL = "Waiting"
DISPLAY_CONF = 0.0

# =========================
# FUNCTIONS
# =========================
def extract_keypoints(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    if not result.pose_landmarks:
        return None

    landmarks = result.pose_landmarks.landmark
    keypoints = []

    for idx in KEYPOINT_IDS:
        lm = landmarks[idx]
        keypoints.extend([lm.x, lm.y])

    return np.array(keypoints, dtype=np.float32)


# =========================
# VIDEO SETUP
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)

fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (w, h)
)

sequence = []

# =========================
# MAIN LOOP
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    kp = extract_keypoints(frame)

    if kp is None:
        PRED_BUFFER.clear()
        out.write(frame)
        continue

    sequence.append(kp)

    if len(sequence) > WINDOW_SIZE:
        sequence.pop(0)

    if len(sequence) == WINDOW_SIZE:
        window = np.expand_dims(sequence, axis=0)

        pred = model.predict(window, verbose=0)
        cls = int(np.argmax(pred))
        conf = float(np.max(pred))

        if conf >= CONF_THRESHOLD:
            PRED_BUFFER.append(cls)

        if len(PRED_BUFFER) == PRED_BUFFER.maxlen:
            stable_cls, count = Counter(PRED_BUFFER).most_common(1)[0]

            if count >= 10:
                DISPLAY_LABEL = id_to_label[stable_cls]
                DISPLAY_CONF = conf

    # =========================
    # DRAW OUTPUT
    # =========================
    cv2.rectangle(frame, (10, 10), (520, 70), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Adavu: {DISPLAY_LABEL}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {DISPLAY_CONF:.2f}",
        (360, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    out.write(frame)

# =========================
# CLEANUP
# =========================
cap.release()
out.release()
pose.close()

print("DONE! Output saved to:", OUTPUT_PATH)
