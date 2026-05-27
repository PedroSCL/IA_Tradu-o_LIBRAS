"""
Etapa 3 — Treinamento do Modelo LSTM
======================================
Lê X.npy e y.npy de data/processed/, constrói e treina uma rede LSTM
para classificação de sinais de Libras.

Uso:
    python src/model/train.py
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CLASSES, NUM_CLASSES,
    DATA_PROCESSED, MODELS_DIR, FIGURES_DIR,
    FRAMES_PER_SEQUENCE, NUM_LANDMARKS_PER_FRAME,
    LSTM_UNITS_1, LSTM_UNITS_2, LSTM_UNITS_3,
    DENSE_UNITS, DROPOUT_RATE, LEARNING_RATE,
    BATCH_SIZE, EPOCHS, TEST_SIZE, RANDOM_STATE,
    MODEL_FILENAME,
)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.utils import to_categorical
except ImportError:
    print("[ERRO] TensorFlow não instalado. Execute: pip install tensorflow")
    sys.exit(1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")   # sem display (servidor / CI)
import matplotlib.pyplot as plt


# ── Reprodutibilidade ─────────────────────────────────────────────────────
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def load_data() -> tuple[np.ndarray, np.ndarray]:
    X_path = os.path.join(DATA_PROCESSED, "X.npy")
    y_path = os.path.join(DATA_PROCESSED, "y.npy")

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print("[ERRO] Dados processados não encontrados.")
        print("       Execute: python src/preprocessing/extract_landmarks.py")
        sys.exit(1)

    X = np.load(X_path)
    y = np.load(y_path)
    print(f"[OK] Dados carregados — X: {X.shape}  y: {y.shape}")
    return X, y


def build_model() -> tf.keras.Model:
    """
    Arquitetura LSTM empilhado para classificação de séries temporais.

    Input:  (FRAMES_PER_SEQUENCE, NUM_LANDMARKS_PER_FRAME)
    Output: (NUM_CLASSES,)  — probabilidades softmax
    """
    model = Sequential([
        LSTM(LSTM_UNITS_1, return_sequences=True,
             input_shape=(FRAMES_PER_SEQUENCE, NUM_LANDMARKS_PER_FRAME),
             name="lstm_1"),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),

        LSTM(LSTM_UNITS_2, return_sequences=True, name="lstm_2"),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),

        LSTM(LSTM_UNITS_3, return_sequences=False, name="lstm_3"),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),

        Dense(DENSE_UNITS, activation="relu", name="dense_1"),
        Dropout(DROPOUT_RATE),

        Dense(NUM_CLASSES, activation="softmax", name="output"),
    ], name="LibrasLSTM")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],    label="Treino")
    axes[0].plot(history.history["val_accuracy"], label="Validação")
    axes[0].set_title("Acurácia por Época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Acurácia")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"],    label="Treino")
    axes[1].plot(history.history["val_loss"], label="Validação")
    axes[1].set_title("Loss por Época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SALVO] Gráfico de treinamento → {save_path}")


def train():
    X, y = load_data()

    # One-hot encoding das labels
    y_cat = to_categorical(y, num_classes=NUM_CLASSES)

    # Divisão treino / teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[OK] Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

    model = build_model()
    model.summary()

    # ── Callbacks ────────────────────────────────────────────────────────
    checkpoint_path = os.path.join(MODELS_DIR, "best_model.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
        ModelCheckpoint(filepath=checkpoint_path, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
    ]

    # ── Treinamento ──────────────────────────────────────────────────────
    print("\n[INÍCIO] Treinamento...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Avaliação no conjunto de teste ───────────────────────────────────
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[RESULTADO] Acurácia no Teste: {acc*100:.2f}%  |  Loss: {loss:.4f}")

    # ── Salvar modelo final ──────────────────────────────────────────────
    final_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    model.save(final_path)
    print(f"[SALVO] Modelo final → {final_path}")

    # ── Salvar histórico como CSV ────────────────────────────────────────
    import pandas as pd
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(MODELS_DIR, "training_history.csv"), index=False)

    # ── Plotar curvas ────────────────────────────────────────────────────
    plot_history(history, os.path.join(FIGURES_DIR, "training_curves.png"))

    print("\n[CONCLUÍDO] Treinamento finalizado!")
    return model, history, X_test, y_test


if __name__ == "__main__":
    train()
