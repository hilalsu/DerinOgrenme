import os
from typing import Dict, List

import numpy as np

from data_loader import load_cifar10_from_disk
from knn import KNearestNeighbor


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Dogru siniflandirma oranini hesaplar."""
    return float(np.mean(y_true == y_pred))


def evaluate_knn_for_k_values(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k_values: List[int],
) -> Dict[int, float]:
    """Her k degeri icin KNN accuracy hesaplar."""
    model = KNearestNeighbor()
    model.fit(X_train, y_train)

    results: Dict[int, float] = {}
    for k in k_values:
        y_pred = model.predict(X_test, k=k)
        acc = accuracy_score(y_test, y_pred)
        results[k] = acc
    return results


def print_results(results: Dict[int, float]) -> None:
    """Sonuclari ekrana yazdirir ve k degerlerini karsilastirir."""
    print("\nKNN Sonuclari (CIFAR-10)")
    print("-" * 35)
    for k, acc in results.items():
        print(f"k = {k:<2} | accuracy = {acc:.4f} ({acc * 100:.2f}%)")

    best_k = max(results, key=results.get)
    print("-" * 35)
    print(f"En iyi sonuc: k = {best_k} -> accuracy = {results[best_k]:.4f}")


def main() -> None:
    # odev1 klasorunden bir ust dizin repo kokudur.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(repo_root, "data")

    # Not: Naive KNN tam CIFAR-10 (50k/10k) ile yavas olabilir.
    # Hiz icin varsayilan alt-kume kullaniliyor.
    X_train, y_train, X_test, y_test = load_cifar10_from_disk(
        data_dir=data_dir,
        train_limit=5000,
        test_limit=1000,
    )

    print(f"Train shape: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Test  shape: {X_test.shape}, Labels: {y_test.shape}")

    k_values = [1, 3, 5, 7]
    results = evaluate_knn_for_k_values(X_train, y_train, X_test, y_test, k_values)
    print_results(results)


if __name__ == "__main__":
    main()
