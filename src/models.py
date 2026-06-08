"""
models.py
---------
Два варианта сети для бинарной классификации:

1. SingleLayerNet  — «однослойная» сеть (нет скрытых слоёв):
       Input → Linear → Sigmoid → output
   Схема: x  →  [W1, b1]  →  sigmoid  →  ŷ

2. OneHiddenLayerNet — перцептрон с одним скрытым слоем:
       Input → Linear → BReLU → Linear → Sigmoid → output
   Схема: x  →  [W1, b1]  →  BReLU  →  [W2, b2]  →  Sigmoid  →  ŷ

Оба класса имеют единый интерфейс:
    forward(X)              → ŷ
    backward(grad_loss)     → обновляет .dW, .db в слоях
    get_params_and_grads()  → список (param, grad) для оптимизатора
    predict(X)              → бинарные метки (порог 0.5)
    predict_proba(X)        → вероятности
"""

import numpy as np
from src.layers import Linear
from src.activations import BReLU, Sigmoid


class SingleLayerNet:
    """
    Сеть без скрытых слоёв: прямая проекция входа на выход через Sigmoid.

    Эквивалентна логистической регрессии.
    Используется как базовая линия.
    """

    def __init__(self, in_features: int, seed: int = 42):
        # Единственный линейный слой: in_features → 1
        self.linear = Linear(in_features, 1, seed=seed)
        self.sigmoid = Sigmoid()

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Прямой проход: (N, in) → (N, 1)"""
        z = self.linear.forward(X)        # (N, 1)
        out = self.sigmoid.forward(z)     # (N, 1)
        return out

    def backward(self, grad_loss: np.ndarray) -> None:
        """
        grad_loss : ∂L/∂ŷ от функции потерь, shape (N, 1)
        Запускает chain rule назад по всем слоям.
        """
        g = self.sigmoid.backward(grad_loss)   # (N, 1)
        self.linear.backward(g)                 # обновляет dW, db

    def get_params_and_grads(self):
        """Возвращает список (param, grad) для оптимизатора."""
        return self.linear.get_params()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int).flatten()


class OneHiddenLayerNet:
    """
    Перцептрон с одним скрытым слоем:
        x → Linear(in, hidden) → BReLU → Linear(hidden, 1) → Sigmoid → ŷ
    """

    def __init__(self, in_features: int, hidden_size: int = 16, seed: int = 42):
        self.linear1 = Linear(in_features, hidden_size, seed=seed)
        self.brelu = BReLU()
        self.linear2 = Linear(hidden_size, 1, seed=seed + 1)
        self.sigmoid = Sigmoid()

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Прямой проход: (N, in) → (N, 1)"""
        z1 = self.linear1.forward(X)       # (N, hidden)
        a1 = self.brelu.forward(z1)        # (N, hidden)
        z2 = self.linear2.forward(a1)      # (N, 1)
        out = self.sigmoid.forward(z2)     # (N, 1)
        return out

    def backward(self, grad_loss: np.ndarray) -> None:
        """Chain rule от выхода к входу."""
        g = self.sigmoid.backward(grad_loss)   # (N, 1)
        g = self.linear2.backward(g)            # (N, hidden)
        g = self.brelu.backward(g)              # (N, hidden)
        self.linear1.backward(g)                # обновляет dW1, db1

    def get_params_and_grads(self):
        """Список всех (param, grad) для передачи оптимизатору."""
        return self.linear1.get_params() + self.linear2.get_params()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int).flatten()
