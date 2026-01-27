import torch
import torch.nn as nn

class TemporalShift(nn.Module):
    """
    Temporal Shift Module (TSM)
    Input:  x [N, T, C, H, W]
    Output: x [N, T, C, H, W] (gleiche Form, teilweise Kanäle zeitlich verschoben)

    fold_div bestimmt, wie viele Kanäle verschoben werden:
      fold = C // fold_div
      - 0..fold-1        : Shift in die Vergangenheit (t -> t+1, also aus t-1 holen)
      - fold..2*fold-1   : Shift in die Zukunft (t -> t-1, also aus t+1 holen)
      - rest             : bleibt gleich
    """
    def __init__(self, n_segment: int, fold_div: int = 8):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, C, H, W]
        assert x.dim() == 5, "TSM erwartet 5D Tensor [N,T,C,H,W]"
        N, T, C, H, W = x.shape
        assert T == self.n_segment, f"T ({T}) muss n_segment ({self.n_segment}) entsprechen"

        fold = C // self.fold_div
        if fold == 0:
            return x  # zu wenige Kanäle, nichts zu verschieben

        out = x.clone()

        # Shift: erste fold Kanäle nach "links" (aus Vergangenheit holen)
        # out[:, t, :fold] = x[:, t-1, :fold]
        out[:, 1:, :fold, :, :] = x[:, :-1, :fold, :, :]
        out[:, 0, :fold, :, :] = 0

        # Shift: nächste fold Kanäle nach "rechts" (aus Zukunft holen)
        # out[:, t, fold:2*fold] = x[:, t+1, fold:2*fold]
        out[:, :-1, fold:2*fold, :, :] = x[:, 1:, fold:2*fold, :, :]
        out[:, -1, fold:2*fold, :, :] = 0

        # Rest bleibt gleich )
        return out
