from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.Introduction.dogs_vs_cats.video_dataset import VideoFrameDataset
from models.Introduction.baseline_seq_cnn import BaselineSeqCNN


def run_one_epoch(model, loader, optimizer, criterion, device, train: bool):
    # Trainings- oder Testmodus setzen
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # beim Testen keine Gradienten berechnen
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for x, y in loader:
            # Daten auf GPU / CPU verschieben
            x = x.to(device)
            y = y.to(device)

            if train:
                optimizer.zero_grad()

            # Vorwärtsdurchlauf
            logits = model(x)
            loss = criterion(logits, y)

            # Rückwärtsdurchlauf nur im Training
            if train:
                loss.backward()
                optimizer.step()

            # Statistiken
            bs = x.size(0)
            total_loss += loss.item() * bs
            correct += (logits.argmax(1) == y).sum().item()
            total += bs

    return total_loss / total, correct / total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # Grundeinstellungen
    T = 8
    batch_size = 4
    epochs = 10
    lr = 1e-3

    # Projektpfade
    project_root = Path(__file__).resolve().parents[2]
    frames_root = project_root / "data" / "frames"

    # Datensätze
    train_ds = VideoFrameDataset(
        frames_root=frames_root,
        T=T,
        split="train",
        train_ratio=0.8,
        samples_per_video=20,
    )

    test_ds = VideoFrameDataset(
        frames_root=frames_root,
        T=T,
        split="test",
        train_ratio=0.8,
        samples_per_video=5,
    )

    # DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("class_to_idx:", train_ds.class_to_idx)

    # Modell (ohne TSM)
    model = BaselineSeqCNN(
        num_classes=2,
        n_segment=T
    ).to(device)

    # Loss-Funktion und Optimierer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Trainingsschleife
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_one_epoch(model, train_loader, optimizer, criterion, device, train=True)
        te_loss, te_acc = run_one_epoch(model, test_loader, optimizer, criterion, device, train=False)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"test loss {te_loss:.4f} acc {te_acc:.3f}"
        )

    # Modell speichern
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "baseline_seq_cnn.pt"
    torch.save(model.state_dict(), out_path)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
