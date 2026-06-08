# src/__init__.py
# Экспортируем основные модули для удобного импорта
from src.activations import BReLU, Sigmoid
from src.losses import BinaryCrossEntropy
from src.layers import Linear
from src.optimizers import AdamOptimizer, SGD
from src.models import SingleLayerNet, OneHiddenLayerNet
from src.trainer import Trainer
from src.data_utils import train_val_test_split, StandardScaler
from src.visualization import (
    plot_history,
    plot_decision_boundary,
    print_metrics,
    plot_confusion_matrix,
)
