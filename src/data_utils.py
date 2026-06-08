"""
data_utils.py
-------------
Утилиты для подготовки данных:
  - train/val/test split (без shuffle — он уже в sklearn)
  - стандартизация (StandardScaler вручную)
  - воспроизводимость через seed
"""

import numpy as np
from typing import Tuple


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, ...]:
    """
    Делит данные на train/val/test в соотношении
        (1 - val_size - test_size) / val_size / test_size.

    Разбиение стратифицировано: сначала отделяем test, потом val от оставшегося.

    Возвращает: X_train, X_val, X_test, y_train, y_val, y_test
    """
    rng = np.random.default_rng(random_state)
    N = X.shape[0]
    idx = rng.permutation(N)

    n_test = int(np.round(N * test_size))
    n_val  = int(np.round(N * val_size))

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test: n_test + n_val]
    train_idx = idx[n_test + n_val:]

    return (
        X[train_idx], X[val_idx], X[test_idx],
        y[train_idx], y[val_idx], y[test_idx],
    )


class StandardScaler:
    """
    Стандартизация признаков:
        X_scaled = (X − μ) / (σ + ε)

    Параметры μ и σ вычисляются только по обучающей выборке
    (во избежание утечки информации из test/val).
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> 'StandardScaler':
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "Вызовите fit() перед transform()"
        return (X - self.mean_) / (self.std_ + self.eps)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
