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

    Das entspricht dem "residual shift"-Gedanken aus dem TSM Paper:
    Der Shift "stört" nicht den Identity-Pfad; räumliche Info bleibt stabil.
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


class TSM_CNN(nn.Module):
    """
    Sauberes Backbone mit echten Residual-Blöcken + drei Köpfen:
    motion (5), rot (3), shape (2)

    Input:  x [N, T, 3, H, W]
    Output: logits_motion [N,5], logits_rot [N,3], logits_shape [N,2]
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
    ):
        super().__init__()
        self.n_segment = n_segment

        self.layer1 = BasicBlockTSM(3,   32, stride=1, n_segment=n_segment, fold_div=fold_div, use_tsm=use_tsm)
        self.layer2 = BasicBlockTSM(32,  64, stride=2, n_segment=n_segment, fold_div=fold_div, use_tsm=use_tsm)
        self.layer3 = BasicBlockTSM(64, 128, stride=2, n_segment=n_segment, fold_div=fold_div, use_tsm=False)       
        self.layer4 = BasicBlockTSM(128,128, stride=1, n_segment=n_segment, fold_div=fold_div, use_tsm=False)

        self.adapt = nn.AdaptiveAvgPool2d((4, 4))
        feat_dim = 128 * 4 * 4

        self.fc_shared = nn.Linear(feat_dim, 256)
        self.drop = nn.Dropout(p=dropout_p)

        self.fc_motion = nn.Linear(256, num_motion_classes)
        self.fc_rot    = nn.Linear(256, num_rot_classes)
        self.fc_shape  = nn.Linear(256, num_shape_classes)

    def forward(self, x: torch.Tensor):
        N, T, C, H, W = x.shape
        assert T == self.n_segment, "Clip-Länge T muss n_segment entsprechen"

        # Frameweise 2D Verarbeitung
        x = x.reshape(N * T, C, H, W)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.adapt(x)
        x = x.reshape(N, T, -1)
        x = x.mean(dim=1)

        x = F.relu(self.fc_shared(x))
        x = self.drop(x)

        return self.fc_motion(x), self.fc_rot(x), self.fc_shape(x)
