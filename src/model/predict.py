import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    CLASSES,
    MODELS_DIR,
    MODEL_FILENAME,
    FRAMES_PER_SEQUENCE,
    NUM_LANDMARKS_PER_FRAME,
    PREDICTION_THRESHOLD
)

# ── MediaPipe ─────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def extract_keypoints(results):

    pose = np.array(
        [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
    ).flatten() if results.pose_landmarks else np.zeros(33 * 3)

    lh = np.array(
        [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
    ).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)

    rh = np.array(
        [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
    ).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([pose, lh, rh])


def draw_landmarks(image, results):

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )


# ── Carregar modelo ───────────────────────────────────────
model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)

if not os.path.exists(model_path):
    print("[ERRO] Modelo não encontrado.")
    sys.exit(1)

model = tf.keras.models.load_model(model_path)

print("[OK] Modelo carregado.")

# ── Procurar câmera ─────────────────────────────────────

camera_index = None

for i in range(10):

    print(f"[TESTANDO] Camera {i}")

    test_cap = cv2.VideoCapture(i)

    if test_cap.isOpened():

        ret, frame = test_cap.read()

        if ret:
            print(f"[OK] Camera encontrada no indice {i}")
            camera_index = i
            test_cap.release()
            break

    test_cap.release()

if camera_index is None:
    print("[ERRO] Nenhuma camera encontrada.")
    sys.exit(1)

cap = cv2.VideoCapture(camera_index)

sequence = deque(maxlen=FRAMES_PER_SEQUENCE)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic.process(image)

        keypoints = extract_keypoints(results)

        sequence.append(keypoints)

        draw_landmarks(frame, results)

        prediction_text = "..."

        if len(sequence) == FRAMES_PER_SEQUENCE:

            input_data = np.expand_dims(sequence, axis=0)

            prediction = model.predict(input_data, verbose=0)[0]

            predicted_class = np.argmax(prediction)

            confidence = prediction[predicted_class]

            print(prediction)

            prediction_text = (
                f"{CLASSES[predicted_class]} "
                f"({confidence:.2f})"
                )

        cv2.rectangle(frame, (0, 0), (640, 40), (0, 0, 0), -1)

        cv2.putText(
            frame,
            prediction_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Tradutor de Libras", frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()