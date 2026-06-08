"""
visualization.py
----------------
Утилиты для визуализации:
  - plot_history     : кривые обучения (loss + accuracy)
  - plot_decision_boundary : граница решения для 2D данных
  - plot_metrics_bar : финальные метрики по тестовой выборке
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, Optional


def plot_history(
    history: Dict[str, list],
    title: str = "Training History",
    figsize: tuple = (12, 4),
):
    """
    Рисует два графика рядом:
    левый  — train/val loss по эпохам
    правый — train/val accuracy по эпохам
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss
    ax = axes[0]
    ax.plot(history['train_loss'], label='Train loss', color='steelblue')
    ax.plot(history['val_loss'],   label='Val loss',   color='tomato', linestyle='--')
    ax.set_title(f"{title} — Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(history['train_acc'], label='Train acc', color='steelblue')
    ax.plot(history['val_acc'],   label='Val acc',   color='tomato', linestyle='--')
    ax.set_title(f"{title} — Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_decision_boundary(
    model,
    X: np.ndarray,
    y: np.ndarray,
    scaler=None,
    title: str = "Decision Boundary",
    figsize: tuple = (7, 5),
    h: float = 0.02,
):
    """
    Отображает границу решения для 2D данных.
    scaler: если передан, применяет transform к сетке перед predict.

    Параметры
    ---------
    model  : обученная модель с методом predict_proba
    X      : данные в оригинальном (не стандартизованном) пространстве
    y      : метки
    scaler : экземпляр StandardScaler (опционально)
    h      : шаг сетки
    """
    assert X.shape[1] == 2, "plot_decision_boundary работает только для 2D данных"

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    if scaler is not None:
        grid = scaler.transform(grid)

    # Вероятности на сетке
    Z = model.predict_proba(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=figsize)
    ax.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.7)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1.5)

    # Точки данных
    scatter = ax.scatter(
        X[:, 0], X[:, 1],
        c=y, cmap='RdBu', edgecolors='k', linewidths=0.5, s=40, alpha=0.85
    )
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    plt.colorbar(scatter, ax=ax, label='Class')
    plt.tight_layout()
    plt.show()


def print_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "Test",
):
    """
    Печатает precision, recall, F1, accuracy для бинарной классификации.

    Реализовано вручную (без sklearn) для наглядности.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    acc       = (TP + TN) / len(y_true)
    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    print(f"\n{'='*40}")
    print(f"  Metrics on {label} set")
    print(f"{'='*40}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  TP={TP}  TN={TN}  FP={FP}  FN={FN}")
    print(f"{'='*40}\n")

    return {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    figsize: tuple = (4, 4),
):
    """Отрисовка матрицы ошибок без sklearn."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    cm = np.array([[TN, FP], [FN, TP]])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred 0', 'Pred 1'])
    ax.set_yticklabels(['True 0', 'True 1'])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)

    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()
