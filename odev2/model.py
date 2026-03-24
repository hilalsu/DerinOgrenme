import torch
import torch.nn as nn


class FashionMNISTClassifier(nn.Module):
    """
    Basit bir yapay sinir agi:
    - Giris: 28x28 = 784
    - 1 hidden layer
    - Cikis: 10 sinif
    """

    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
