from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.Introduction.dogs_vs_cats.video_dataset import VideoFrameDataset
from models.Introduction.tsm_cnn_old import TSM_CNN


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
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(1) == y).sum().item()
            total += batch_size

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    # Gerät auswählen
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
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    print("class_to_idx:", train_ds.class_to_idx)

    # Modell
    model = TSM_CNN(
        num_classes=2,
        n_segment=T
    ).to(device)

    # Loss-Funktion und Optimierer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Trainingsschleife
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            train=True,
        )

        test_loss, test_acc = run_one_epoch(
            model,
            test_loader,
            optimizer,
            criterion,
            device,
            train=False,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"test loss {test_loss:.4f} acc {test_acc:.3f}"
        )


if __name__ == "__main__":
    main()
