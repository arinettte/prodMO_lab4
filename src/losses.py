"""
losses.py
---------
Функция потерь Binary Cross-Entropy (BCE):

    L = -1/N * Σ [y_i * log(ŷ_i) + (1 − y_i) * log(1 − ŷ_i)]

Где:
    y_i  — истинная метка (0 или 1)
    ŷ_i  — предсказанная вероятность (выход сигмоиды, ∈ (0, 1))

Градиент по ŷ_i:
    ∂L/∂ŷ_i = (ŷ_i − y_i) / (ŷ_i * (1 − ŷ_i))
"""

import numpy as np


class BinaryCrossEntropy:
    """
    Binary Cross-Entropy loss.

    Внимание: при совместном использовании с Sigmoid на выходном слое
    градиент ∂L/∂z (до сигмоиды) упрощается до (ŷ − y)/N.
    Здесь реализован полный вариант: градиент по выходу сигмоиды,
    а упрощение возникает автоматически при chain rule через Sigmoid.backward().
    """

    def __init__(self, eps: float = 1e-12):
        # eps предотвращает log(0)
        self.eps = eps

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Вычисляет среднее значение BCE по батчу.
        y_pred : (N,) или (N,1) — предсказания после Sigmoid
        y_true : (N,) или (N,1) — истинные метки {0, 1}
        """
        # Сохраняем для backward
        self.y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        self.y_true = y_true.reshape(y_pred.shape)
        N = y_pred.shape[0]
        loss = -np.mean(
            self.y_true * np.log(self.y_pred) +
            (1.0 - self.y_true) * np.log(1.0 - self.y_pred)
        )
        return float(loss)

    def backward(self) -> np.ndarray:
        """
        Градиент BCE по y_pred (выходу сигмоиды):
            ∂L/∂ŷ = -(y/ŷ - (1-y)/(1-ŷ)) / N
        """
        N = self.y_pred.shape[0]
        grad = -(self.y_true / self.y_pred - (1.0 - self.y_true) / (1.0 - self.y_pred)) / N
        return grad
