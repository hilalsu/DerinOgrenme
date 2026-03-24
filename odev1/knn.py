from typing import Optional

import numpy as np


class KNearestNeighbor:
    """
    Basit KNN siniflandiricisi (from scratch).
    sklearn kullanilmadan numpy ile yazilmistir.
    """

    def __init__(self) -> None:
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train verisini ezberler (KNN'in egitim adimi)."""
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray, k: int = 1) -> np.ndarray:
        """Verilen X icin k en yakin komsuya gore tahmin uretir."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model once fit edilmelidir.")
        if k <= 0:
            raise ValueError("k degeri pozitif olmalidir.")
        if k > self.X_train.shape[0]:
            raise ValueError("k, train ornek sayisindan buyuk olamaz.")

        dists = self._compute_distances_no_loops(X)
        return self._predict_labels(dists, k)

    def _compute_distances_no_loops(self, X: np.ndarray) -> np.ndarray:
        """
        Tum test-train ciftleri icin Oklid uzakligini vektorel hesaplar.
        Donus: (num_test, num_train)
        """
        if self.X_train is None:
            raise ValueError("Model once fit edilmelidir.")

        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        X_sq = np.sum(X**2, axis=1, keepdims=True)  # (num_test, 1)
        train_sq = np.sum(self.X_train**2, axis=1)  # (num_train,)
        cross = X @ self.X_train.T  # (num_test, num_train)

        dists_sq = X_sq + train_sq - 2 * cross
        # Sayisal hatalardan dolayi kucuk negatif degerler olusabilir
        dists_sq = np.maximum(dists_sq, 0.0)
        dists = np.sqrt(dists_sq)
        return dists

    def _predict_labels(self, dists: np.ndarray, k: int) -> np.ndarray:
        """Mesafe matrisinden cogunluk oylamasi ile sinif tahmin eder."""
        if self.y_train is None:
            raise ValueError("Model once fit edilmelidir.")

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test, dtype=np.int64)

        for i in range(num_test):
            # En kucuk k uzakligin indexlerini al
            k_nearest_idx = np.argpartition(dists[i], k - 1)[:k]
            k_nearest_labels = self.y_train[k_nearest_idx]

            # En sik sinifi sec (beraberlikte kucuk sinif etiketi kazanir)
            y_pred[i] = np.bincount(k_nearest_labels).argmax()

        return y_pred
