"""
trainer.py
----------
Цикл обучения с поддержкой mini-batch SGD.

Функциональность:
  - mini-batch SGD с произвольным batch_size
  - логирование train/val loss и accuracy каждую эпоху
  - Early Stopping по val_loss (patience + min_delta)
  - сохранение лучших весов (in-memory snapshot)
  - опциональное перемешивание данных каждую эпоху
"""

import numpy as np
import copy
from typing import Tuple, Dict, Any

from src.losses import BinaryCrossEntropy


class Trainer:
    """
    Универсальный тренер для моделей из models.py.

    Параметры
    ---------
    model      : экземпляр SingleLayerNet или OneHiddenLayerNet
    optimizer  : экземпляр AdamOptimizer или SGD
    batch_size : размер мини-батча (при batch_size=None — полный батч)
    epochs     : максимальное число эпох
    patience   : сколько эпох ждать улучшения val_loss (Early Stopping)
    min_delta  : минимальное улучшение, считающееся значимым
    shuffle    : перемешивать ли данные перед каждой эпохой
    verbose    : печатать ли прогресс
    """

    def __init__(
        self,
        model,
        optimizer,
        batch_size: int = 32,
        epochs: int = 200,
        patience: int = 20,
        min_delta: float = 1e-4,
        shuffle: bool = True,
        verbose: bool = True,
        verbose_every: int = 10,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = BinaryCrossEntropy()
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.min_delta = min_delta
        self.shuffle = shuffle
        self.verbose = verbose
        self.verbose_every = verbose_every

        # История обучения
        self.history: Dict[str, list] = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }

        # Лучшие веса (deep copy состояний слоёв)
        self._best_weights = None
        self._best_val_loss = np.inf
        self._no_improve_count = 0

    def _accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Вычисляет accuracy (доля верных ответов)."""
        preds = (y_pred >= 0.5).astype(int).flatten()
        return float(np.mean(preds == y_true.flatten()))

    def _run_epoch(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Один проход по обучающим данным.
        Возвращает (средний loss, accuracy) по эпохе.
        """
        N = X.shape[0]
        indices = np.arange(N)

        if self.shuffle:
            np.random.shuffle(indices)

        batch_size = self.batch_size if self.batch_size is not None else N

        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        # Итерируемся по мини-батчам
        for start in range(0, N, batch_size):
            idx = indices[start: start + batch_size]
            X_batch = X[idx]
            y_batch = y[idx]

            # --- Forward ---
            y_pred = self.model.forward(X_batch)            # (bs, 1)

            # --- Loss ---
            loss = self.criterion.forward(y_pred, y_batch)

            # --- Backward ---
            grad_loss = self.criterion.backward()            # ∂L/∂ŷ
            self.model.backward(grad_loss)

            # --- Optimizer step ---
            params_grads = self.model.get_params_and_grads()
            self.optimizer.step(params_grads)

            acc = self._accuracy(y_pred, y_batch)
            epoch_loss += loss
            epoch_acc += acc
            n_batches += 1

        return epoch_loss / n_batches, epoch_acc / n_batches

    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Оценка модели без обновления весов."""
        y_pred = self.model.forward(X)
        loss = self.criterion.forward(y_pred, y)
        acc = self._accuracy(y_pred, y)
        return loss, acc

    def _snapshot_weights(self):
        """Сохраняет копию текущих весов модели."""
        # Deep copy всех параметров через get_params_and_grads
        snapshot = []
        for param, _ in self.model.get_params_and_grads():
            snapshot.append(param.copy())
        return snapshot

    def _restore_weights(self, snapshot):
        """Восстанавливает веса из снимка."""
        for (param, _), saved in zip(self.model.get_params_and_grads(), snapshot):
            param[:] = saved

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, list]:
        """
        Основной цикл обучения.

        Параметры
        ---------
        X_train, y_train : обучающая выборка
        X_val, y_val     : валидационная выборка

        Возвращает историю обучения.
        """
        # Сброс оптимизатора и истории при повторном обучении
        self.optimizer.reset()
        self.history = {k: [] for k in self.history}
        self._best_val_loss = np.inf
        self._no_improve_count = 0
        self._best_weights = None

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self._run_epoch(X_train, y_train)
            val_loss, val_acc = self._evaluate(X_val, y_val)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            # --- Early Stopping ---
            if val_loss < self._best_val_loss - self.min_delta:
                self._best_val_loss = val_loss
                self._best_weights = self._snapshot_weights()
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1

            if self.verbose and epoch % self.verbose_every == 0:
                print(
                    f"Epoch {epoch:4d}/{self.epochs} | "
                    f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
                    f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f}"
                )

            if self._no_improve_count >= self.patience:
                if self.verbose:
                    print(f"\nEarly stopping at epoch {epoch}.")
                break

        # Восстанавливаем лучшие веса
        if self._best_weights is not None:
            self._restore_weights(self._best_weights)
            if self.verbose:
                print(f"Best val loss: {self._best_val_loss:.4f} — weights restored.")

        return self.history
