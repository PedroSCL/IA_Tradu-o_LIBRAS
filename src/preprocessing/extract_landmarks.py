"""
Etapa 2 — Pré-processamento
============================
Lê os arquivos .npy coletados em data/raw, monta o dataset (X, y)
e salva em data/processed/ para uso no treinamento.

Uso:
    python src/preprocessing/extract_landmarks.py
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CLASSES, DATA_RAW, DATA_PROCESSED,
    SEQUENCES_PER_CLASS, FRAMES_PER_SEQUENCE,
    NUM_LANDMARKS_PER_FRAME,
)


def load_dataset() -> tuple[np.ndarray, np.ndarray]:

    X, y = [], []

    label_map = {cls: idx for idx, cls in enumerate(CLASSES)}

    print("[INFO] Carregando dados brutos...")

    for cls in CLASSES:

        cls_path = os.path.join(DATA_RAW, cls)

        if not os.path.exists(cls_path):
            continue

        cls_ok = 0

        sequences = sorted([
            d for d in os.listdir(cls_path)
            if os.path.isdir(os.path.join(cls_path, d))
        ])

        for seq in sequences:

            seq_path = os.path.join(
                cls_path,
                seq,
                "keypoints.npy"
            )

            if not os.path.exists(seq_path):
                print(f"  [AVISO] Arquivo não encontrado: {seq_path}")
                continue

            frames = np.load(seq_path)

            # Reamostragem uniforme para exatamente 30 frames

            if len(frames) == 0:
                continue

            indices = np.linspace(
                0,
                len(frames) - 1,
                FRAMES_PER_SEQUENCE
            ).astype(int)

            frames = frames[indices]

            X.append(frames)
            y.append(label_map[cls])

            cls_ok += 1

        print(f"  {cls:<20} → {cls_ok} sequências carregadas")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return X, y

def save_processed(X: np.ndarray, y: np.ndarray):
    """Salva arrays processados e um CSV com metadados."""
    np.save(os.path.join(DATA_PROCESSED, "X.npy"), X)
    np.save(os.path.join(DATA_PROCESSED, "y.npy"), y)

    # CSV de metadados para facilitar análise exploratória
    meta = pd.DataFrame({
        "sample_idx": range(len(y)),
        "label_idx":  y,
        "label_name": [CLASSES[i] for i in y],
    })
    meta.to_csv(os.path.join(DATA_PROCESSED, "metadata.csv"), index=False)

    print(f"\n[SALVO] X.npy   → shape {X.shape}")
    print(f"[SALVO] y.npy   → shape {y.shape}")
    print(f"[SALVO] metadata.csv → {len(meta)} amostras")


def print_summary(X: np.ndarray, y: np.ndarray):
    print("\n─── Resumo do Dataset ───────────────────────────────")
    print(f"  Total de amostras   : {len(X)}")
    print(f"  Shape de X          : {X.shape}  (amostras × frames × features)")
    print(f"  Distribuição de classes:")
    for cls_idx, cls in enumerate(CLASSES):
        count = int((y == cls_idx).sum())
        print(f"    [{cls_idx}] {cls:<12} : {count} amostras")
    print("─────────────────────────────────────────────────────")


if __name__ == "__main__":
    X, y = load_dataset()

    if len(X) == 0:
        print("\n[ERRO] Nenhum dado encontrado em data/raw/")
        print("       Execute primeiro: python src/data_collection/collect_data.py")
        sys.exit(1)

    print_summary(X, y)
    save_processed(X, y)
    print("\n[CONCLUÍDO] Pré-processamento finalizado!")
