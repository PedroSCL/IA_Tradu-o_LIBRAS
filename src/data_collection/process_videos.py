import os
import cv2
import numpy as np
import mediapipe as mp
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    CLASSES,
    DATA_RAW,
    FRAMES_PER_SEQUENCE,
    NUM_LANDMARKS_PER_FRAME
)

mp_holistic = mp.solutions.holistic


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


with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    for action in CLASSES:

        action_path = os.path.join(DATA_RAW, action)

        videos = [
            video for video in os.listdir(action_path)
            if video.endswith(".mp4")
        ]

        print(f"\nClasse: {action}")

        for idx, video_name in enumerate(videos):

            video_path = os.path.join(action_path, video_name)

            print(f"Processando: {video_name}")

            cap = cv2.VideoCapture(video_path)

            sequence = []

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                image = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                results = holistic.process(image)

                keypoints = extract_keypoints(results)

                sequence.append(keypoints)

            cap.release()

            # Limitar quantidade de frames
            if len(sequence) > FRAMES_PER_SEQUENCE:
                sequence = sequence[:FRAMES_PER_SEQUENCE]

            # Completar frames faltantes
            while len(sequence) < FRAMES_PER_SEQUENCE:
                sequence.append(
                    np.zeros(NUM_LANDMARKS_PER_FRAME)
                )

            sequence = np.array(sequence)

            save_dir = os.path.join(
                DATA_RAW,
                action,
                str(idx)
            )

            os.makedirs(save_dir, exist_ok=True)

            np.save(
                os.path.join(save_dir, "keypoints.npy"),
                sequence
            )

            print(f"Salvo em: {save_dir}")

print("\nProcessamento concluído.")