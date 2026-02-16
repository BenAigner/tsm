import torch
import torch.nn as nn
import torch.nn.functional as F

from models.simpleForms.tsm import TemporalShift


class TSM_CNN(nn.Module):
    """
    Sequenz-CNN mit Temporal Shift Module (TSM).

    Eingabe:
        x: [N, T, 3, H, W]

    Unterschied zur Baseline:
        - zusätzliche zeitliche Informationsmischung durch TSM
        - keine 3D-Convolutions notwendig
        - gleiche Grundarchitektur für fairen Vergleich
    """

    def __init__(self, num_classes: int = 10, n_segment: int = 4, fold_div: int = 8):
        super().__init__()
        self.n_segment = n_segment

        # Feature-Extraktion pro Frame
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)

        # zeitliche Verschiebung nach der ersten Convolution
        self.tsm1 = TemporalShift(n_segment=n_segment, fold_div=fold_div)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)

        # zweite zeitliche Verschiebung mit höherer Kanalanzahl
        self.tsm2 = TemporalShift(n_segment=n_segment, fold_div=fold_div)

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

        # erste Convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Rückformung für zeitliche Verschiebung
        x = x.reshape(N, T, 16, H, W)
        x = self.tsm1(x)

        # erneut flach für weitere 2D-Operationen
        x = x.reshape(N * T, 16, H, W)

        # zweite Convolution
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # zweite zeitliche Verschiebung
        x = x.reshape(N, T, 32, H, W)
        x = self.tsm2(x)

        # wieder flach für Pooling
        x = x.reshape(N * T, 32, H, W)

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
