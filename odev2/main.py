import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import create_data_loaders, load_fashion_mnist_from_disk
from model import FashionMNISTClassifier


def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Batch accuracy hesaplar."""
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Modeli bir epoch egitir ve ortalama loss/accuracy dondurur."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_acc = calculate_accuracy(logits, labels)
        total_loss += loss.item()
        total_acc += batch_acc
        total_batches += 1

    return total_loss / total_batches, total_acc / total_batches


def evaluate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Test verisinde loss ve accuracy hesaplar."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            batch_acc = calculate_accuracy(logits, labels)
            total_loss += loss.item()
            total_acc += batch_acc
            total_batches += 1

    return total_loss / total_batches, total_acc / total_batches


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanilan cihaz: {device}")

    # Repo kokunden data/fashion-mnist klasorunu hedefler.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(repo_root, "data", "fashion-mnist")

    X_train, y_train, X_test, y_test = load_fashion_mnist_from_disk(data_dir)
    train_loader, test_loader = create_data_loaders(
        X_train, y_train, X_test, y_test, batch_size=128
    )

    model = FashionMNISTClassifier(input_size=784, hidden_size=128, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%"
        )

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print("\nTest Sonuclari")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
