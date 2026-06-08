"""
optimizers.py
-------------
Реализация SGD с поддержкой mini-batch и Adam-модификации.

Adam = Momentum (или Nesterov) + RMSProp (или AdaGrad) + bias correction.

Математика:

--- SGD с momentum ---
    v_t     = β * v_{t-1} + (1 − β) * g_t
    θ_{t+1} = θ_t − α * v_t

--- Nesterov (NAG) ---
    v_t     = β * v_{t-1} + g(θ_t − β * v_{t-1})   ← look-ahead
    (В матричной реализации аппроксимируется через:)
    v_t     = β * v_{t-1} + g_t
    θ_{t+1} = θ_t − α * (β * v_t + g_t)

--- AdaGrad ---
    G_t     = G_{t-1} + g_t²
    θ_{t+1} = θ_t − (α / sqrt(G_t + ε)) * g_t

--- RMSProp ---
    s_t     = ρ * s_{t-1} + (1 − ρ) * g_t²
    θ_{t+1} = θ_t − (α / sqrt(s_t + ε)) * g_t

--- Adam (Momentum + RMSProp + bias correction) ---
    m_t     = β1 * m_{t-1} + (1 − β1) * g_t          ← 1-й момент
    v_t     = β2 * v_{t-1} + (1 − β2) * g_t²         ← 2-й момент
    m̂_t    = m_t / (1 − β1^t)                         ← коррекция смещения
    v̂_t    = v_t / (1 − β2^t)                         ← коррекция смещения
    θ_{t+1} = θ_t − α * m̂_t / (sqrt(v̂_t) + ε)
"""

import numpy as np
from typing import List, Tuple


class AdamOptimizer:
    """
    Универсальный оптимизатор с настраиваемыми компонентами:

    momentum_type  : 'momentum' | 'nesterov'
    adaptive_type  : 'adagrad'  | 'rmsprop'

    При momentum_type='momentum' и adaptive_type='rmsprop' — стандартный Adam.
    При momentum_type='nesterov' — Nadam-подобное поведение.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,       # коэффициент momentum / Nesterov
        beta2: float = 0.999,     # коэффициент для AdaGrad / RMSProp
        eps: float = 1e-8,        # числовая стабильность
        weight_decay: float = 0.0,  # L2-регуляризация
        momentum_type: str = 'momentum',   # 'momentum' или 'nesterov'
        adaptive_type: str = 'rmsprop',    # 'adagrad' или 'rmsprop'
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum_type = momentum_type
        self.adaptive_type = adaptive_type

        # Состояния: для каждого параметра (индексируем по id объекта)
        self.m = {}  # первый момент (momentum)
        self.v = {}  # второй момент (адаптивный)
        self.t = 0   # счётчик шагов (для bias correction)

    def step(self, params_and_grads: List[Tuple[np.ndarray, np.ndarray]]):
        """
        params_and_grads : список пар (W, dW) из каждого слоя.
        Обновляет веса in-place.
        """
        self.t += 1

        for param, grad in params_and_grads:
            if grad is None:
                continue

            pid = id(param)

            # Инициализируем состояния при первом обращении к параметру
            if pid not in self.m:
                self.m[pid] = np.zeros_like(param)
                self.v[pid] = np.zeros_like(param)

            g = grad.copy()

            # L2-регуляризация (weight decay): добавляем λ*θ к градиенту
            if self.weight_decay > 0.0:
                g += self.weight_decay * param

            # --- Momentum / Nesterov ---
            self.m[pid] = self.beta1 * self.m[pid] + (1.0 - self.beta1) * g

            if self.momentum_type == 'nesterov':
                # Nesterov: look-ahead через β1*m + (1-β1)*g
                m_hat_num = self.beta1 * self.m[pid] + (1.0 - self.beta1) * g
            else:
                # Стандартный momentum
                m_hat_num = self.m[pid]

            # Коррекция смещения для первого момента
            m_hat = m_hat_num / (1.0 - self.beta1 ** self.t)

            # --- AdaGrad / RMSProp ---
            if self.adaptive_type == 'adagrad':
                # AdaGrad: аккумулируем все прошлые квадраты (без decay)
                self.v[pid] += g ** 2
                v_hat = self.v[pid] / (1.0 - self.beta2 ** self.t)
                # Замечание: bias correction для AdaGrad нестандартно,
                # но сохраняем для симметрии; v_hat растёт монотонно
            else:
                # RMSProp: экспоненциальное скользящее среднее квадратов
                self.v[pid] = self.beta2 * self.v[pid] + (1.0 - self.beta2) * g ** 2
                v_hat = self.v[pid] / (1.0 - self.beta2 ** self.t)

            # --- Обновление параметра ---
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self):
        """Сброс состояний (при повторном обучении)."""
        self.m = {}
        self.v = {}
        self.t = 0


class SGD:
    """
    Классический SGD (без адаптивных методов) — для сравнения.

    θ_{t+1} = θ_t − α * g_t
    """

    def __init__(self, lr: float = 1e-2, weight_decay: float = 0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, params_and_grads: List[Tuple[np.ndarray, np.ndarray]]):
        for param, grad in params_and_grads:
            if grad is None:
                continue
            g = grad.copy()
            if self.weight_decay > 0.0:
                g += self.weight_decay * param
            param -= self.lr * g

    def reset(self):
        pass
