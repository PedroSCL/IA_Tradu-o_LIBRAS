import cv2
import numpy as np
import mediapipe as mp
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import DATA_RAW

VIDEOS_DIR = os.path.join("data", "videos")

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


videos = [v for v in os.listdir(VIDEOS_DIR) if v.endswith(".mp4")]

print(f"[INFO] {len(videos)} vídeos encontrados")


with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    for idx, video_name in enumerate(videos):

        print(f"\n[{idx+1}/{len(videos)}] Processando: {video_name}")

        label = video_name.split("_")[0].lower()

        video_path = os.path.join(VIDEOS_DIR, video_name)

        cap = cv2.VideoCapture(video_path)

        frames_data = []

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = holistic.process(image)

            keypoints = extract_keypoints(results)

            frames_data.append(keypoints)

        cap.release()

        save_dir = os.path.join(DATA_RAW, label, str(idx))

        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, "keypoints.npy")

        np.save(
            save_path,
            np.array(frames_data, dtype=np.float32)
        )

        print(f"[SALVO] {save_path}")

print("\n[CONCLUÍDO] Todos os vídeos foram processados.")