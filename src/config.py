"""
config.py — Configurações globais do projeto Tradutor de Libras
"""

import os

# ── Diretórios ─────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW      = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
REPORTS_DIR   = os.path.join(BASE_DIR, "reports")
FIGURES_DIR   = os.path.join(REPORTS_DIR, "figures")

for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Sinais / Classes ────────────────────────────────────────────────────────
CLASSES = sorted([
    d for d in os.listdir(DATA_RAW)
    if os.path.isdir(os.path.join(DATA_RAW, d))
])

NUM_CLASSES = len(CLASSES)

# ── Coleta de dados ─────────────────────────────────────────────────────────
SEQUENCES_PER_CLASS = 30   # quantas sequências (vídeos) por sinal
FRAMES_PER_SEQUENCE = 60   # frames por sequência

# ── MediaPipe ───────────────────────────────────────────────────────────────
# Cada frame produz: 21 landmarks mão esquerda + 21 mão direita + 33 pose = 75
# Cada landmark tem (x, y, z) → 75 * 3 = 225 features por frame
NUM_LANDMARKS_PER_FRAME = 225

# ── Modelo LSTM ─────────────────────────────────────────────────────────────
LSTM_UNITS_1   = 32
LSTM_UNITS_2   = 64
LSTM_UNITS_3   = 32
DENSE_UNITS    = 32
DROPOUT_RATE   = 0.4
LEARNING_RATE  = 1e-3
BATCH_SIZE     = 8
EPOCHS         = 100
TEST_SIZE      = 0.20
RANDOM_STATE   = 42
MODEL_FILENAME = "libras_lstm.keras"

# ── Interface ───────────────────────────────────────────────────────────────
PREDICTION_THRESHOLD = 0.95  # confiança mínima para aceitar predição
TTS_LANG             = "pt"   # idioma do gTTS
