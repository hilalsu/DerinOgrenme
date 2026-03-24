import gzip
import os
import urllib.request
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


FASHION_MNIST_URLS = {
    "train_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
    "train_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
    "test_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
    "test_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
}


def _download_if_missing(data_dir: str) -> None:
    """Fashion-MNIST dosyalari yoksa indirir."""
    os.makedirs(data_dir, exist_ok=True)
    for url in FASHION_MNIST_URLS.values():
        filename = os.path.basename(url)
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            print(f"Indiriliyor: {filename}")
            urllib.request.urlretrieve(url, file_path)


def _load_idx_images(path: str) -> np.ndarray:
    """IDX image dosyasini (gzip) numpy array olarak okur."""
    with gzip.open(path, "rb") as f:
        # IDX format: magic(4) + count(4) + rows(4) + cols(4)
        data = f.read()
    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    images = images.reshape(-1, 28 * 28)
    return images


def _load_idx_labels(path: str) -> np.ndarray:
    """IDX label dosyasini (gzip) numpy array olarak okur."""
    with gzip.open(path, "rb") as f:
        # IDX format: magic(4) + count(4)
        data = f.read()
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    return labels


def load_fashion_mnist_from_disk(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fashion-MNIST verisini diskten okur, normalize eder ve train/test dondurur.
    """
    _download_if_missing(data_dir)

    train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    test_images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")

    X_train = _load_idx_images(train_images_path).astype(np.float32) / 255.0
    y_train = _load_idx_labels(train_labels_path).astype(np.int64)
    X_test = _load_idx_images(test_images_path).astype(np.float32) / 255.0
    y_test = _load_idx_labels(test_labels_path).astype(np.int64)

    return X_train, y_train, X_test, y_test


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 128,
) -> Tuple[DataLoader, DataLoader]:
    """Numpy dizilerinden PyTorch DataLoader olusturur."""
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
