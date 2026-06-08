"""
layers.py
---------
Полносвязный (линейный) слой:

    z = X @ W + b

    X : (N, in_features)
    W : (in_features, out_features)
    b : (1, out_features)
    z : (N, out_features)

Градиенты (по chain rule):
    ∂L/∂W = X^T @ δ          (in_features × out_features)
    ∂L/∂b = sum(δ, axis=0)   (1 × out_features)
    ∂L/∂X = δ @ W^T          (N × in_features)

где δ = ∂L/∂z — градиент, пришедший «сверху».
"""

import numpy as np


class Linear:
    """
    Линейный слой: z = X @ W + b

    Параметры инициализируются методом He (для ReLU-подобных активаций):
        W ~ N(0, sqrt(2 / in_features))
    b = 0
    """

    def __init__(self, in_features: int, out_features: int, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        # He-инициализация: уменьшает риск затухания/взрыва градиентов
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))

        # Градиенты — заполним при первом backward
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Имя слоя для оптимизатора
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Сохраняем вход для backward
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        grad_output : ∂L/∂z  (N × out_features)
        Возвращает ∂L/∂X    (N × in_features)
        """
        N = self.input.shape[0]
        # Градиенты по весам и смещениям
        self.dW = self.input.T @ grad_output          # (in × out)
        self.db = np.sum(grad_output, axis=0, keepdims=True)  # (1 × out)
        # Градиент, передаваемый ниже по сети
        return grad_output @ self.W.T                 # (N × in)

    def get_params(self):
        """Возвращает параметры и их градиенты в виде списка пар (param, grad)."""
        return [(self.W, self.dW), (self.b, self.db)]
