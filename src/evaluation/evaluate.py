"""
Etapa 4 — Avaliação do Modelo
===============================
Carrega o modelo treinado e os dados de teste, calcula métricas completas
(acurácia, precisão, recall, F1) e gera:
  - Matriz de confusão
  - Relatório de classificação
  - Gráfico de métricas por classe

Uso:
    python src/evaluation/evaluate.py
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CLASSES, NUM_CLASSES,
    DATA_PROCESSED, MODELS_DIR, FIGURES_DIR,
    TEST_SIZE, RANDOM_STATE, MODEL_FILENAME,
)

try:
    import tensorflow as tf
except ImportError:
    print("[ERRO] TensorFlow não instalado.")
    sys.exit(1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)
from tensorflow.keras.utils import to_categorical

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def load_test_data():
    X = np.load(os.path.join(DATA_PROCESSED, "X.npy"))
    y = np.load(os.path.join(DATA_PROCESSED, "y.npy"))

    y_cat = to_categorical(y, num_classes=NUM_CLASSES)
    _, X_test, _, y_test = train_test_split(
        X, y_cat, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    _, _, _, y_test_raw = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_test, y_test, y_test_raw


def plot_confusion_matrix(cm: np.ndarray, save_path: str):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES,
        linewidths=0.5,
    )
    plt.title("Matriz de Confusão — Tradutor de Libras", fontsize=14, fontweight="bold")
    plt.ylabel("Rótulo Real")
    plt.xlabel("Rótulo Predito")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SALVO] Matriz de confusão → {save_path}")


def plot_metrics_per_class(report: dict, save_path: str):
    classes_in_report = [c for c in CLASSES if c in report]
    metrics = ["precision", "recall", "f1-score"]

    x = np.arange(len(classes_in_report))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        values = [report[cls][metric] for cls in classes_in_report]
        ax.bar(x + i * width, values, width, label=metric.capitalize())

    ax.set_xticks(x + width)
    ax.set_xticklabels(classes_in_report)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Métricas por Classe de Sinal", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SALVO] Métricas por classe → {save_path}")


def evaluate():
    # ── Carregar modelo ────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    if not os.path.exists(model_path):
        # Tenta best_model como fallback
        model_path = os.path.join(MODELS_DIR, "best_model.keras")
        if not os.path.exists(model_path):
            print("[ERRO] Nenhum modelo treinado encontrado em models/")
            print("       Execute: python src/model/train.py")
            sys.exit(1)

    print(f"[OK] Carregando modelo: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # ── Carregar dados de teste ────────────────────────────────────────────
    X_test, y_test_cat, y_test_raw = load_test_data()
    print(f"[OK] Amostras de teste: {len(X_test)}")

    # ── Predições ──────────────────────────────────────────────────────────
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)

    # ── Métricas globais ───────────────────────────────────────────────────
    acc  = accuracy_score(y_test_raw, y_pred)
    prec = precision_score(y_test_raw, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test_raw, y_pred,    average="weighted", zero_division=0)
    f1   = f1_score(y_test_raw, y_pred,        average="weighted", zero_division=0)

    print("\n════════════════════════════════════════════")
    print("          RESULTADOS — MODELO LSTM          ")
    print("════════════════════════════════════════════")
    print(f"  Acurácia  : {acc*100:6.2f}%")
    print(f"  Precisão  : {prec*100:6.2f}%")
    print(f"  Recall    : {rec*100:6.2f}%")
    print(f"  F1-Score  : {f1*100:6.2f}%")
    print("════════════════════════════════════════════\n")

    # ── Relatório por classe ───────────────────────────────────────────────
    report = classification_report(
        y_test_raw, y_pred,
        target_names=CLASSES, output_dict=True, zero_division=0,
    )
    print(classification_report(y_test_raw, y_pred, target_names=CLASSES, zero_division=0))

    # ── Matriz de confusão ─────────────────────────────────────────────────
    cm = confusion_matrix(y_test_raw, y_pred)
    plot_confusion_matrix(cm, os.path.join(FIGURES_DIR, "confusion_matrix.png"))

    # ── Métricas por classe ────────────────────────────────────────────────
    plot_metrics_per_class(report, os.path.join(FIGURES_DIR, "metrics_per_class.png"))

    # ── Salvar resumo em TXT ──────────────────────────────────────────────
    summary_path = os.path.join(FIGURES_DIR, "evaluation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Avaliação do Modelo LSTM — Tradutor de Libras ===\n\n")
        f.write(f"Acurácia  : {acc*100:.2f}%\n")
        f.write(f"Precisão  : {prec*100:.2f}%\n")
        f.write(f"Recall    : {rec*100:.2f}%\n")
        f.write(f"F1-Score  : {f1*100:.2f}%\n\n")
        f.write(classification_report(y_test_raw, y_pred, target_names=CLASSES, zero_division=0))
    print(f"[SALVO] Resumo da avaliação → {summary_path}")

    print("\n[CONCLUÍDO] Avaliação finalizada!")
    return acc


if __name__ == "__main__":
    evaluate()
