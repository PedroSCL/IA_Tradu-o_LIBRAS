import cv2
import numpy as np
import mediapipe as mp
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import DATA_RAW

mp_holistic = mp.solutions.holistic


def extract_keypoints(results):

    pose = (
        np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.pose_landmarks.landmark]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(99)
    )

    lh = (
        np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(63)
    )

    rh = (
        np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(63)
    )

    return np.concatenate([pose, lh, rh])


classes = [
    d for d in os.listdir(DATA_RAW)
    if os.path.isdir(os.path.join(DATA_RAW, d))
]

print(f"[INFO] Classes encontradas: {classes}")

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    for label in classes:

        class_dir = os.path.join(DATA_RAW, label)

        videos = [
            v for v in os.listdir(class_dir)
            if v.lower().endswith((".mp4", ".avi", ".mov"))
        ]

        print(f"\n[{label}] {len(videos)} vídeos encontrados")

        for idx, video_name in enumerate(videos):

            video_path = os.path.join(class_dir, video_name)

            cap = cv2.VideoCapture(video_path)

            frames_data = []

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = holistic.process(image)

                frames_data.append(
                    extract_keypoints(results)
                )

            cap.release()

            save_dir = os.path.join(class_dir, str(idx))
            os.makedirs(save_dir, exist_ok=True)

            np.save(
                os.path.join(save_dir, "keypoints.npy"),
                np.array(frames_data, dtype=np.float32)
            )

            print(
                f"[{idx+1}/{len(videos)}] "
                f"{video_name} -> {len(frames_data)} frames"
            )

print("\n[CONCLUÍDO] Todos os vídeos foram processados.")