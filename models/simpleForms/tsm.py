import torch
import torch.nn as nn

class TemporalShift(nn.Module):
    """
    Temporal Shift Module (TSM)
    Input : x [N, T, C, H, W]
    Output: y [N, T, C, H, W]

    fold = C // fold_div
      - Kanäle [0:fold]        -> Shift aus Vergangenheit: y[t] = x[t-1]
      - Kanäle [fold:2*fold]   -> Shift aus Zukunft:      y[t] = x[t+1]
      - Rest                   -> bleibt gleich

    Performance:
      - Kein in-place auf x (autograd-sicher).
      - Kein x.clone() (vermeidet volle Kopie). Wir schreiben gezielt in ein neues Tensor.
    """
    def __init__(self, n_segment: int, fold_div: int = 8):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5, "TSM erwartet [N,T,C,H,W]"
        N, T, C, H, W = x.shape
        assert T == self.n_segment, f"T={T} muss n_segment={self.n_segment} sein"

        fold = C // self.fold_div
        if fold == 0:
            return x

        y = x.new_zeros((N, T, C, H, W))

        # Rest: unverändert kopieren
        y[:, :, 2*fold:, :, :] = x[:, :, 2*fold:, :, :]

        # Past shift: y[t] bekommt x[t-1]
        y[:, 1:, :fold, :, :] = x[:, :-1, :fold, :, :]

        # Future shift: y[t] bekommt x[t+1]
        y[:, :-1, fold:2*fold, :, :] = x[:, 1:, fold:2*fold, :, :]

        return y

