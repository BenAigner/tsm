import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineSeqCNN(nn.Module):
    """
    Einfaches Sequenz-CNN ohne zeitliche Modellierung.

    Eingabe:
        x: [N, T, 3, H, W]

    Vorgehen:
        - jedes Frame wird unabhängig mit einem 2D-CNN verarbeitet
        - pro Frame entsteht ein Feature-Vektor
        - zeitliche Aggregation erfolgt über Mittelwertbildung
        - Klassifikation über Fully-Connected-Layer
    """

    def __init__(self, num_classes: int = 10, n_segment: int = 4):
        super().__init__()
        self.n_segment = n_segment

        # Convolutional Feature-Extraktion pro Frame
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)

        # räumliches Downsampling
        self.pool = nn.MaxPool2d(2, 2)

        # vereinheitlicht die Featuregröße unabhängig von der Eingangsauflösung
        self.adapt = nn.AdaptiveAvgPool2d((8, 8))

        # Klassifikationskopf
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, C, H, W]
        N, T, C, H, W = x.shape
        assert T == self.n_segment

        # Zeitdimension in die Batch-Dimension falten
        x = x.reshape(N * T, C, H, W)

        # Feature-Extraktion
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # räumliche Reduktion
        x = self.pool(x)
        x = self.pool(x)

        # feste Featuregröße erzeugen
        x = self.adapt(x)

        # pro Frame flatten
        x = x.reshape(N, T, -1)

        # zeitliche Aggregation
        x = x.mean(dim=1)

        # Klassifikation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
