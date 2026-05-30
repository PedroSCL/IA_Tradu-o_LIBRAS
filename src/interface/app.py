"""
Etapas 5 + 6 — Interface em Tempo Real + Síntese de Voz
=========================================================
Abre a câmera, extrai landmarks com MediaPipe, classifica com o modelo
LSTM e converte o resultado em áudio (TTS) automaticamente.

Controles:
    Q  — sair
    R  — resetar histórico de frames
    ESPAÇO — forçar fala do sinal atual

Uso:
    python src/interface/app.py
"""

import cv2
import numpy as np
import os
import sys
import time
from collections import deque, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CLASSES, MODELS_DIR, MODEL_FILENAME,
    FRAMES_PER_SEQUENCE, NUM_LANDMARKS_PER_FRAME,
    PREDICTION_THRESHOLD, TTS_LANG,
)

# ── Importações opcionais ──────────────────────────────────────────────────────
try:
    import mediapipe as mp
except ImportError:
    print("[ERRO] mediapipe não instalado: pip install mediapipe")
    sys.exit(1)

try:
    import tensorflow as tf
except ImportError:
    print("[ERRO] tensorflow não instalado: pip install tensorflow")
    sys.exit(1)

# TTS — tenta pyttsx3 (offline) e cai para gTTS (online)
TTS_ENGINE = None
try:
    import pyttsx3
    TTS_ENGINE = "pyttsx3"
except ImportError:
    pass

if TTS_ENGINE is None:
    try:
        from gtts import gTTS
        import subprocess
        TTS_ENGINE = "gtts"
    except ImportError:
        print("[AVISO] Nenhum motor TTS disponível. Áudio desativado.")


# ── MediaPipe setup ────────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

SPOKEN_TEXT = {
    "amarelo": "Amarelo",
    "banco": "Banco",
    "espelho": "Espelho",
}

# ── Parâmetros anti-falso-positivo ─────────────────────────────────────────────
VOTES_REQUIRED  = 3    # sinal precisa aparecer N vezes seguidas para ser aceito
MIN_HAND_FRAMES = 40    # mínimo de frames com mão detectada no buffer para predizer


def speak(text: str):
    if TTS_ENGINE is None:
        return
    if TTS_ENGINE == "pyttsx3":
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        for v in voices:
            if "brazil" in v.id.lower() or "pt" in v.id.lower():
                engine.setProperty("voice", v.id)
                break
        engine.setProperty("rate", 150)
        engine.say(text)
        engine.runAndWait()
    elif TTS_ENGINE == "gtts":
        tts = gTTS(text=text, lang=TTS_LANG)
        tmp = os.path.join(os.environ.get("TEMP", "/tmp"), "libras_tts.mp3")
        tts.save(tmp)
        os.system(f'start /min wmplayer "{tmp}"' if sys.platform == "win32"
                  else f"mpg123 {tmp} > /dev/null 2>&1")


def extract_keypoints(results) -> np.ndarray:
    pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
                    dtype=np.float32).flatten() if results.pose_landmarks else np.zeros(99, dtype=np.float32)
    lh   = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
                    dtype=np.float32).flatten() if results.left_hand_landmarks else np.zeros(63, dtype=np.float32)
    rh   = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
                    dtype=np.float32).flatten() if results.right_hand_landmarks else np.zeros(63, dtype=np.float32)
    return np.concatenate([pose, lh, rh]), (results.left_hand_landmarks is not None or
                                             results.right_hand_landmarks is not None)


def draw_landmarks(image, results):
    opts_lm = mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3)
    opts_cn = mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS, opts_lm, opts_cn)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS, opts_lm, opts_cn)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                  mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=3),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2))


def draw_ui(frame, current_word, confidence, history, hand_detected, hand_frames):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "TRADUTOR DE LIBRAS - IA",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "[Q] Sair  [R] Reset  [ESPACO] Falar",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Indicador de mão detectada
    hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    hand_text  = f"Maos: {'DETECTADAS' if hand_detected else 'NAO DETECTADAS'}  ({hand_frames}/{FRAMES_PER_SEQUENCE} frames)"
    cv2.putText(frame, hand_text, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)

    if current_word:
        label = SPOKEN_TEXT.get(current_word, current_word)
        cv2.putText(frame, f"> {label}",
                    (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
        cv2.putText(frame, f"Confianca: {confidence*100:.1f}%",
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Aguardando sinal...",
                    (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 1)

    hist_str = "  |  ".join(history[-5:]) if history else "—"
    cv2.putText(frame, f"Historico: {hist_str}",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return frame


def run():
    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, "best_model.keras")
        if not os.path.exists(model_path):
            print("[ERRO] Modelo não encontrado. Execute primeiro: python src/model/train.py")
            sys.exit(1)

    print(f"[OK] Carregando modelo: {model_path}")
    model = tf.keras.models.load_model(model_path)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERRO] Câmera não disponível.")
        sys.exit(1)

    frame_buffer  = deque(maxlen=FRAMES_PER_SEQUENCE)
    hand_buffer   = deque(maxlen=FRAMES_PER_SEQUENCE)  # rastreia se havia mão em cada frame
    vote_buffer   = deque(maxlen=VOTES_REQUIRED)        # últimas N predições para votação
    word_history  = []
    current_word  = ""
    confidence    = 0.0
    last_spoken   = ""
    last_speak_t  = 0.0
    SPEAK_COOLDOWN = 4.0
    hand_detected  = False

    print("[INÍCIO] Interface iniciada! Pressione Q para sair.\n")

    with mp_holistic.Holistic(min_detection_confidence=0.6,
                              min_tracking_confidence=0.6) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            draw_landmarks(frame, results)

            kp, hand_now = extract_keypoints(results)
            frame_buffer.append(kp)
            hand_buffer.append(hand_now)
            hand_detected = hand_now

            hand_frames = sum(hand_buffer)  # quantos frames do buffer têm mão

            # ── Predição com filtros anti-falso-positivo ──────────────────────
            if len(frame_buffer) == FRAMES_PER_SEQUENCE:

                # FILTRO 1: exige mínimo de frames com mão detectada
                if hand_frames >= MIN_HAND_FRAMES:
                    seq   = np.expand_dims(np.array(frame_buffer, dtype=np.float32), axis=0)
                    probs = model.predict(seq, verbose=0)[0]
                    print("\nProbabilidades:")
                    for i, p in enumerate(probs):
                        print(f"{CLASSES[i]}: {p:.4f}")
                    pred_idx   = int(np.argmax(probs))
                    pred_conf  = float(probs[pred_idx])

                    if pred_conf >= PREDICTION_THRESHOLD:
                        vote_buffer.append(pred_idx)

                        # FILTRO 2: votação — aceita só se N predições seguidas concordam
                        if (len(vote_buffer) == VOTES_REQUIRED and
                                len(set(vote_buffer)) == 1):

                            confidence   = pred_conf
                            current_word = CLASSES[pred_idx]

                            now = time.time()
                            if (current_word != last_spoken and
                                    now - last_speak_t > SPEAK_COOLDOWN):
                                text_to_speak = SPOKEN_TEXT.get(current_word, current_word)
                                print(f"[SINAL] {current_word}  ({confidence*100:.1f}%)  → '{text_to_speak}'")
                                speak(text_to_speak)
                                word_history.append(current_word)
                                last_spoken  = current_word
                                last_speak_t = now
                                frame_buffer.clear()
                                hand_buffer.clear()
                                vote_buffer.clear()
                    else:
                        vote_buffer.clear()
                else:
                    # Sem mãos suficientes — limpa votos e predição
                    vote_buffer.clear()
                    current_word = ""
                    confidence   = 0.0

            frame = draw_ui(frame, current_word, confidence, word_history,
                            hand_detected, hand_frames)
            cv2.imshow("Tradutor de Libras — IA", frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                frame_buffer.clear()
                hand_buffer.clear()
                vote_buffer.clear()
                current_word = ""
                confidence   = 0.0
                print("[RESET] Buffer limpo.")
            elif key == ord(" ") and current_word:
                speak(SPOKEN_TEXT.get(current_word, current_word))

    cap.release()
    cv2.destroyAllWindows()
    print("[FIM] Interface encerrada.")


if __name__ == "__main__":
    run()