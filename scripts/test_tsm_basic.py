import torch
from models.tsm import TemporalShift

def main():
    # Kleine, übersichtliche Größen
    N = 1
    T = 4
    C = 4
    H = W = 1

    # Tensor mit klaren Werten: Wert = t*10 + c
    x = torch.zeros((N, T, C, H, W), dtype=torch.int64)
    for t in range(T):
        for c in range(C):
            x[0, t, c, 0, 0] = t * 10 + c

    print("Input (Frame x Channel):")
    print(x[0, :, :, 0, 0])  # Form: [T, C]

    # fold_div=2 -> fold = C//2 = 2
    # Kanäle 0-1: links (Vergangenheit)
    # Kanäle 2-3: rechts (Zukunft)
    tsm = TemporalShift(n_segment=T, fold_div=2)

    with torch.no_grad():
        y = tsm(x)

    print("\nOutput (Frame x Channel):")
    print(y[0, :, :, 0, 0])

    # Erwartung grob prüfen: Shape gleich
    assert y.shape == x.shape, "Output-Shape muss gleich Input-Shape sein!"
    print("\n✅ TSM Dummy-Test bestanden (Shape ok).")

if __name__ == "__main__":
    main()
