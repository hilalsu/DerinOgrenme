import os
import pickle
from typing import Optional, Tuple

import numpy as np


def _resolve_cifar10_folder(data_dir: str) -> str:
    """
    CIFAR-10'un diskteki pickle klasorunu bulur.
    Beklenen klasor: data/cifar-10-batches-py
    """
    cifar_dir = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.isdir(cifar_dir):
        raise FileNotFoundError(
            "CIFAR-10 klasoru bulunamadi.\n"
            f"Beklenen konum: {cifar_dir}\n"
            "Lutfen CIFAR-10 Python version dosyasini indirip bu klasore cikarin:\n"
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        )
    return cifar_dir


def _load_pickle_batch(batch_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tek bir CIFAR-10 pickle batch dosyasini okur.
    Donus:
      - X: (N, 3072) uint8
      - y: (N,) int64
    """
    with open(batch_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    X = batch[b"data"]  # (N, 3072)
    y = np.array(batch[b"labels"], dtype=np.int64)
    return X, y


def load_cifar10_from_disk(
    data_dir: str,
    train_limit: Optional[int] = 5000,
    test_limit: Optional[int] = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    CIFAR-10'u diskteki pickle dosyalarindan okuyup train/test dondurur.
    Istege bagli olarak hiz icin train_limit/test_limit ile orneklem alir.
    """
    cifar_dir = _resolve_cifar10_folder(data_dir)

    # Train: data_batch_1..5
    train_batches = []
    train_labels = []
    for i in range(1, 6):
        batch_path = os.path.join(cifar_dir, f"data_batch_{i}")
        X_part, y_part = _load_pickle_batch(batch_path)
        train_batches.append(X_part)
        train_labels.append(y_part)

    X_train = np.concatenate(train_batches, axis=0)
    y_train = np.concatenate(train_labels, axis=0)

    # Test: test_batch
    test_path = os.path.join(cifar_dir, "test_batch")
    X_test, y_test = _load_pickle_batch(test_path)

    # KNN mesafe hesabi icin float'a cevir
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Hiz icin istege bagli alt-kume
    if train_limit is not None:
        X_train = X_train[:train_limit]
        y_train = y_train[:train_limit]
    if test_limit is not None:
        X_test = X_test[:test_limit]
        y_test = y_test[:test_limit]

    return X_train, y_train, X_test, y_test
