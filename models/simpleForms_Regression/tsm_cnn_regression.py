import torch
import torch.nn as nn
import torch.nn.functional as F

from models.simpleForms.tsm import TemporalShift


class BasicBlockTSM(nn.Module):
    """
    ResNet-ähnlicher Block, bei dem TSM im Residual-Branch liegt.

    identity ---------------> (+) -> ReLU
        |                     ^
        v                     |
      (branch) ---------------
      TSM -> Conv -> BN -> ReLU -> Conv -> BN
    """
    def __init__(self, in_ch, out_ch, stride, n_segment, fold_div, use_tsm=True):
        super().__init__()
        self.n_segment = n_segment

        self.tsm = TemporalShift(n_segment=n_segment, fold_div=fold_div) if use_tsm else nn.Identity()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.down = nn.Identity()

    def _apply_tsm(self, x_4d: torch.Tensor) -> torch.Tensor:
        """
        x_4d: [N*T, C, H, W] -> [N, T, C, H, W] -> TSM -> [N*T, C, H, W]
        """
        NT, C, H, W = x_4d.shape
        T = self.n_segment
        assert NT % T == 0, f"NT={NT} ist nicht durch T={T} teilbar"
        N = NT // T

        x_5d = x_4d.reshape(N, T, C, H, W)
        x_5d = self.tsm(x_5d)
        return x_5d.reshape(N * T, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.down(x)

        out = self._apply_tsm(x)  # Shift im Branch
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))

        out = F.relu(out + identity)
        return out


class TSM_CNN_Regression(nn.Module):
    """
    Backbone wie TSM_CNN, aber Multi-Task:
      - Klassifikation: motion (5), rot (3), shape (2)
      - Regression: speed, omega_mag

    Input:  x [N, T, 3, H, W]
    Output: logits_motion [N,5], logits_rot [N,3], logits_shape [N,2],
            pred_speed [N], pred_omega [N]

    Wenn normalize_regression=True (Targets in [0,1]):
      -> pred_speed/pred_omega via sigmoid ebenfalls in (0,1)
    Sonst:
      -> Softplus (>=0, unbounded)
    """
    def __init__(
        self,
        num_motion_classes=5,
        num_rot_classes=3,
        num_shape_classes=2,
        n_segment=16,
        fold_div=8,
        use_tsm=True,
        dropout_p=0.2,
        normalize_regression: bool = False,
    ):
        super().__init__()
        self.n_segment = n_segment
        self.normalize_regression = bool(normalize_regression)

        self.layer1 = BasicBlockTSM(3,   32, stride=1, n_segment=n_segment, fold_div=fold_div, use_tsm=use_tsm)
        self.layer2 = BasicBlockTSM(32,  64, stride=2, n_segment=n_segment, fold_div=fold_div, use_tsm=use_tsm)
        self.layer3 = BasicBlockTSM(64, 128, stride=2, n_segment=n_segment, fold_div=fold_div, use_tsm=use_tsm)
        self.layer4 = BasicBlockTSM(128,128, stride=1, n_segment=n_segment, fold_div=fold_div, use_tsm=use_tsm)

        self.adapt = nn.AdaptiveAvgPool2d((4, 4))
        feat_dim = 128 * 4 * 4

        self.fc_shared = nn.Linear(feat_dim, 256)
        self.drop = nn.Dropout(p=dropout_p)

        # Klassifikations-Heads
        self.fc_motion = nn.Linear(256, num_motion_classes)
        self.fc_rot    = nn.Linear(256, num_rot_classes)
        self.fc_shape  = nn.Linear(256, num_shape_classes)

        # Regressions-Heads
        self.fc_speed = nn.Linear(256, 1)
        self.fc_omega = nn.Linear(256, 1)

        self.softplus = nn.Softplus(beta=1.0, threshold=20.0)

    def forward(self, x: torch.Tensor):
        N, T, C, H, W = x.shape
        assert T == self.n_segment, "Clip-Länge T muss n_segment entsprechen"

        # Frameweise 2D Verarbeitung: [N*T, C, H, W]
        x = x.reshape(N * T, C, H, W)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Spatial pooling -> [N*T, feat_dim]
        x = self.adapt(x)
        x = x.reshape(N, T, -1)
        x = x.mean(dim=1)  # temporal pooling -> [N, feat_dim]

        # Shared MLP -> [N, 256]
        x = F.relu(self.fc_shared(x))
        x = self.drop(x)

        # Heads
        logits_motion = self.fc_motion(x)
        logits_rot    = self.fc_rot(x)
        logits_shape  = self.fc_shape(x)

        raw_speed = self.fc_speed(x).squeeze(1)  # [N]
        raw_omega = self.fc_omega(x).squeeze(1)  # [N]

        if self.normalize_regression:
            pred_speed = torch.sigmoid(raw_speed)  # (0,1)
            pred_omega = torch.sigmoid(raw_omega)  # (0,1)
        else:
            pred_speed = self.softplus(raw_speed)  # >=0
            pred_omega = self.softplus(raw_omega)  # >=0

        return logits_motion, logits_rot, logits_shape, pred_speed, pred_omega
