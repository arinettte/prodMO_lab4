"""
activations.py
--------------
Функции активации, используемые в сети:
  - BReLU (Bipolar ReLU): f(x) = x если x > 0, иначе -x (т.е. |x|)
    Часто трактуется как зеркальная ReLU — допускает отрицательные значения.
    Здесь реализована именно «двусторонняя» версия:
        forward:  max(x, 0)  на «положительной» части  +  min(x, 0) передаётся как есть
    На самом деле классический BReLU = f(x) = x при x≥0, -αx при x<0 (leaky).
    Мы используем вариант: f(x) = x при x≥0, -x при x<0  → |x| (absolute value ReLU),
    что даёт ненулевые градиенты по всей числовой оси.

  - Sigmoid: σ(x) = 1 / (1 + exp(-x)), используется на выходном слое для бинарной классификации.
"""

import numpy as np


class BReLU:
    """
    Bipolar ReLU (абсолютно-значная версия):
        forward:  f(x) = |x|
        backward: f'(x) = sign(x)  (субградиент в 0 = 0)

    Это симметричная функция — не насыщается ни в + ни в −,
    что помогает избежать "мёртвых нейронов" классической ReLU.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Сохраняем вход для обратного прохода
        self.input = x
        return np.abs(x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Производная |x| = sign(x); при x=0 субградиент = 0
        grad = np.sign(self.input)
        return grad_output * grad


class Sigmoid:
    """
    Сигмоида: σ(x) = 1 / (1 + exp(-x))
    Производная: σ'(x) = σ(x) * (1 - σ(x))

    Используется на выходном слое для получения вероятности принадлежности к классу 1.
    Clip применяется для численной стабильности.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Clipping предотвращает overflow в exp
        x_clipped = np.clip(x, -500, 500)
        self.output = 1.0 / (1.0 + np.exp(-x_clipped))
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # σ'(x) = σ(x)(1 − σ(x))
        grad = self.output * (1.0 - self.output)
        return grad_output * grad
