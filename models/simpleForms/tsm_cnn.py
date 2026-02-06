import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tsm import TemporalShift


class ConvBNReLU(nn.Module):
    """
    Standard-Baustein: 2D-Convolution + BatchNorm + ReLU.
    Optionales Downsampling über stride.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class TSM_CNN(nn.Module):
    """
    2D-CNN pro Frame für räumliche Merkmale + optionales Temporal Shift Module (TSM)
    für leichte zeitliche Kopplung. Multi-Task-Ausgabe über drei Köpfe:
      - Motion: 5 Klassen (none, left, right, up, down)
      - Rotation: 3 Klassen (ccw, cw, none)
      - Shape: 2 Klassen (rect, tri)  

    Eingabe:
        x: [N, T, 3, H, W]

    Ausgabe:
        logits_motion: [N, 5]
        logits_rot   : [N, 3]

    Design:
        - Backbone ist bewusst klein, aber tiefer als die 2-Conv-Variante.
        - Downsampling wird früh eingesetzt (stride=2), um das rezeptive Feld zu vergrößern.
        - TSM kann über use_tsm deaktiviert werden; damit erhält man eine faire Baseline
          mit identischer Architektur und Parameteranzahl (abgesehen von TSM selbst).
    """

    def __init__(
        self,
        num_motion_classes: int = 5,
        num_rot_classes: int = 3,
        num_shape_classes: int = 2,
        n_segment: int = 16,
        fold_div: int = 8,
        use_tsm: bool = True,
        dropout_p: float = 0.2
    ):
        super().__init__()
        self.n_segment = n_segment
        self.use_tsm = use_tsm

        # Stage 1: frühe Kanten/Flächenmerkmale
        self.b1 = ConvBNReLU(3, 32, stride=1)
        self.tsm1 = TemporalShift(n_segment=n_segment, fold_div=fold_div) if use_tsm else nn.Identity()

        # Stage 2: Downsampling (rezeptives Feld wird deutlich größer)
        self.b2 = ConvBNReLU(32, 64, stride=2)
        self.tsm2 = TemporalShift(n_segment=n_segment, fold_div=fold_div) if use_tsm else nn.Identity()

        # Stage 3: weitere Feature-Stufe, robustere Orientierungssensitivität
        self.b3 = ConvBNReLU(64, 128, stride=2)
        self.tsm3 = TemporalShift(n_segment=n_segment, fold_div=fold_div) if use_tsm else nn.Identity()

        # Stage 4: zusätzliche Verfeinerung ohne weiteres Downsampling
        self.b4 = ConvBNReLU(128, 128, stride=1)

        # Feste Featuregröße; 4x4 ist ausreichend und reduziert Parameter im Head
        self.adapt = nn.AdaptiveAvgPool2d((4, 4))

        feat_dim = 128 * 4 * 4

        # Gemeinsamer Feature-Head
        self.fc_shared = nn.Linear(feat_dim, 256)
        self.drop = nn.Dropout(p=dropout_p)

        # Task-spezifische Ausgaben
        self.fc_motion = nn.Linear(256, num_motion_classes)
        self.fc_rot = nn.Linear(256, num_rot_classes)
        self.fc_shape = nn.Linear(256, num_shape_classes)

    def _apply_tsm(self, x: torch.Tensor, tsm: nn.Module, C: int, H: int, W: int) -> torch.Tensor:
        """
        Hilfsfunktion: [N*T, C, H, W] -> [N, T, C, H, W] -> TSM -> zurück.
        """
        N_times_T = x.shape[0]
        T = self.n_segment
        N = N_times_T // T

        x = x.reshape(N, T, C, H, W)
        x = tsm(x)
        x = x.reshape(N * T, C, H, W)
        return x

    def forward(self, x: torch.Tensor):
        N, T, C, H, W = x.shape
        assert T == self.n_segment, "Die Clip-Länge T muss n_segment entsprechen"

        # [N,T,C,H,W] -> [N*T,C,H,W] für 2D-Verarbeitung pro Frame
        x = x.reshape(N * T, C, H, W)

        # Stage 1
        x = self.b1(x)  # [N*T, 32, H, W]
        x = self._apply_tsm(x, self.tsm1, C=32, H=H, W=W)

        # Stage 2 (Downsampling)
        x = self.b2(x)  # [N*T, 64, H/2, W/2]
        H2, W2 = H // 2, W // 2
        x = self._apply_tsm(x, self.tsm2, C=64, H=H2, W=W2)

        # Stage 3 (Downsampling)
        x = self.b3(x)  # [N*T, 128, H/4, W/4]
        H4, W4 = H2 // 2, W2 // 2
        x = self._apply_tsm(x, self.tsm3, C=128, H=H4, W=W4)

        # Stage 4
        x = self.b4(x)  # [N*T, 128, H/4, W/4]

        # Pooling auf feste Größe
        x = self.adapt(x)  # [N*T, 128, 4, 4]

        # pro Frame flatten -> [N, T, F]
        x = x.reshape(N, T, -1)

        # Zeitliche Aggregation (einfache Baseline-Aggregation)
        x = x.mean(dim=1)  # [N, F]

        # Gemeinsame Repräsentation
        x = F.relu(self.fc_shared(x))
        x = self.drop(x)

        # Task-Logits
        logits_motion = self.fc_motion(x)
        logits_rot = self.fc_rot(x)
        logits_shape = self.fc_shape(x)

        return logits_motion, logits_rot, logits_shape
