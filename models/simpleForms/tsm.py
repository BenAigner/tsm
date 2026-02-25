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

    Optimierung (ohne Funktionsänderung):
      - Vermeidet new_zeros über den ganzen Tensor (teure Full-Memory-Init).
      - Nutzt new_empty + setzt nur die Randframes der geshifteten Kanäle auf 0.
       
    """
    def __init__(self, n_segment: int, fold_div: int = 8):
        super().__init__()
        self.n_segment = int(n_segment)
        self.fold_div = int(fold_div)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"TSM erwartet [N,T,C,H,W], bekam shape={tuple(x.shape)}")

        N, T, C, H, W = x.shape
        if T != self.n_segment:
            raise ValueError(f"T={T} muss n_segment={self.n_segment} sein")

        fold = C // self.fold_div
        if fold == 0:
            return x

        # Statt full zero-init: schneller (spart Memory-Bandwidth)
        y = x.new_empty((N, T, C, H, W))

        # Rest: unverändert kopieren
        y[:, :, 2 * fold :, :, :] = x[:, :, 2 * fold :, :, :]

        # Past shift: y[t] = x[t-1] für die ersten fold Kanäle
        y[:, 1:, :fold, :, :] = x[:, :-1, :fold, :, :]
        y[:, 0, :fold, :, :].zero_()  # Randframe entspricht new_zeros Verhalten

        # Future shift: y[t] = x[t+1] für die nächsten fold Kanäle
        y[:, :-1, fold : 2 * fold, :, :] = x[:, 1:, fold : 2 * fold, :, :]
        y[:, -1, fold : 2 * fold, :, :].zero_()  # Randframe entspricht new_zeros Verhalten

        return y
