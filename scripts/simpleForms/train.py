import os
import argparse
import time
from dataclasses import dataclass

# Für das Speichern der finalen Testmetriken und der Confusion Matrix
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Dataset liefert Clips + drei Labels (motion, rotation, shape)
from scripts.simpleForms.synth_dataset import SynthMultiLabelNPZ
# Modell gibt drei Logit-Tensoren zurück: motion, rotation, shape
from models.simpleForms.tsm_cnn import TSM_CNN


@dataclass
class Metrics:
    """
    Container für Metriken pro Epoche/Split.
    acc_exact ist hier als Exact-Match über alle drei Teilaufgaben definiert:
    (motion korrekt) AND (rotation korrekt) AND (shape korrekt).
    """
    loss: float
    acc_motion: float
    acc_rot: float
    acc_shape: float
    acc_exact: float


def set_seed(seed: int) -> None:
    """
    Setzt Zufallsseeds für Reproduzierbarkeit des Trainings.
    Hinweis: DataLoader-Worker haben eigene RNG-Zustände; für viele Experimente reicht
    diese Einstellung aus, insbesondere bei synthetischen Daten.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinismus ist für Vergleichsexperimente hilfreich; kann Training verlangsamen.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             ce_motion: nn.Module,
             ce_rot: nn.Module,
             ce_shape: nn.Module) -> Metrics:
    """
    Evaluation ohne Gradientenberechnung (schneller und speichersparend).
    Berechnet Loss und Accuracies über den gesamten Loader (val oder test).
    """
    model.eval()

    total_loss = 0.0
    correct_m = 0
    correct_r = 0
    correct_s = 0
    correct_exact = 0
    total = 0

    for clips, motion, rot, shape in loader:
        # clips: [B, T, C, H, W]
        clips = clips.to(device)
        motion = motion.to(device)  # [B]
        rot = rot.to(device)        # [B]
        shape = shape.to(device)    # [B]

        # Vorwärtslauf: drei Logit-Ausgaben
        logits_m, logits_r, logits_s = model(clips)

        # Multi-Task-Loss: Summe aus drei Cross-Entropy-Terms
        loss = (
            ce_motion(logits_m, motion)
            + ce_rot(logits_r, rot)
            + ce_shape(logits_s, shape)
        )

        # Klassenentscheidungen als Argmax über Logits
        pred_m = logits_m.argmax(dim=1)
        pred_r = logits_r.argmax(dim=1)
        pred_s = logits_s.argmax(dim=1)

        bsz = clips.size(0)
        total += bsz

        # Loss über Batchgröße gewichtet aufsummieren (für Mittelwert am Ende)
        total_loss += float(loss.item()) * bsz

        # Einzel-Task Treffer
        correct_m += int((pred_m == motion).sum().item())
        correct_r += int((pred_r == rot).sum().item())
        correct_s += int((pred_s == shape).sum().item())

        # Exact-Match: alle drei Labels gleichzeitig korrekt
        correct_exact += int(((pred_m == motion) & (pred_r == rot) & (pred_s == shape)).sum().item())

    # Rückgabe als Mittelwerte (durch total normalisiert)
    return Metrics(
        loss=total_loss / max(total, 1),
        acc_motion=correct_m / max(total, 1),
        acc_rot=correct_r / max(total, 1),
        acc_shape=correct_s / max(total, 1),
        acc_exact=correct_exact / max(total, 1),
    )


@torch.no_grad()
def collect_rotation_preds(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Sammelt True-/Pred-Labels für die Rotationsklassifikation, um eine Confusion Matrix
    auf dem Testset zu erstellen.
    """
    model.eval()
    y_true, y_pred = [], []

    for clips, motion, rot, shape in loader:
        clips = clips.to(device)
        rot = rot.to(device)

        # Nur Rotation wird hier ausgewertet, daher reicht pred_r
        logits_m, logits_r, logits_s = model(clips)
        pred_r = logits_r.argmax(dim=1)

        y_true.extend(rot.cpu().numpy().tolist())
        y_pred.extend(pred_r.cpu().numpy().tolist())

    return y_true, y_pred


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    device: torch.device,
                    optimizer: torch.optim.Optimizer,
                    ce_motion: nn.Module,
                    ce_rot: nn.Module,
                    ce_shape: nn.Module) -> Metrics:
    """
    Führt genau eine Trainingsepoche aus (ein vollständiger Durchlauf über dl_train).
    """
    model.train()

    total_loss = 0.0
    correct_m = 0
    correct_r = 0
    correct_s = 0
    correct_exact = 0
    total = 0

    for clips, motion, rot, shape in loader:
        clips = clips.to(device)
        motion = motion.to(device)
        rot = rot.to(device)
        shape = shape.to(device)

        # Gradienten löschen (set_to_none=True kann etwas effizienter sein)
        optimizer.zero_grad(set_to_none=True)

        # Vorwärtslauf
        logits_m, logits_r, logits_s = model(clips)

        # Multi-Task-Loss als Summe der drei Teilaufgaben
        loss = (
            ce_motion(logits_m, motion)
            + ce_rot(logits_r, rot)
            + ce_shape(logits_s, shape)
        )

        # Backprop + Optimizer-Step
        loss.backward()
        optimizer.step()

        # Predictions pro Task
        pred_m = logits_m.argmax(dim=1)
        pred_r = logits_r.argmax(dim=1)
        pred_s = logits_s.argmax(dim=1)

        bsz = clips.size(0)
        total += bsz
        total_loss += float(loss.item()) * bsz

        # Accuracies
        correct_m += int((pred_m == motion).sum().item())
        correct_r += int((pred_r == rot).sum().item())
        correct_s += int((pred_s == shape).sum().item())

        # Exact-Match über alle drei Tasks
        correct_exact += ((pred_m == motion) & (pred_r == rot) & (pred_s == shape)).sum().item()

    return Metrics(
        loss=total_loss / max(total, 1),
        acc_motion=correct_m / max(total, 1),
        acc_rot=correct_r / max(total, 1),
        acc_shape=correct_s / max(total, 1),
        acc_exact=correct_exact / max(total, 1),
    )


def main():
    """
    Entry-Point: Argumente parsen, Seeds setzen, Loader/Modell initialisieren,
    Training durchführen, bestes Checkpoint speichern und am Ende Test + Confusion Matrix erzeugen.
    """
    ap = argparse.ArgumentParser()

    # Schaltet Temporal Shift im Modell an/aus (Baseline vs. TSM)
    ap.add_argument("--use_tsm", action="store_true", help="TSM aktivieren")

    # Datensatzroot enthält train/val/test Unterordner
    ap.add_argument("--data", type=str, required=True,
                    help="Pfad zum Datensatz-Root, z.B. data/synth_shapes_ml")

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123)

    # Clip-Länge T muss mit n_segment übereinstimmen (TSM erwartet feste Segmentzahl)
    ap.add_argument("--n_segment", type=int, default=16,
                    help="Muss mit T im Datensatz übereinstimmen")
    ap.add_argument("--fold_div", type=int, default=8)

    ap.add_argument("--device", type=str, default="cuda",
                    help="cuda oder cpu")
    ap.add_argument("--save_dir", type=str, default="runs",
                    help="Output-Verzeichnis für Checkpoints/Logs")

    args = ap.parse_args()

    set_seed(args.seed)

    # Fallback auf CPU, falls CUDA nicht verfügbar ist
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Datasets / Loader
    ds_train = SynthMultiLabelNPZ(args.data, "train")
    ds_val = SynthMultiLabelNPZ(args.data, "val")
    ds_test = SynthMultiLabelNPZ(args.data, "test")

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # Modell initialisieren (drei Köpfe: motion, rotation, shape)
    model = TSM_CNN(
        num_motion_classes=5,
        num_rot_classes=3,
        num_shape_classes=2,
        n_segment=args.n_segment,
        fold_div=args.fold_div,
        use_tsm=args.use_tsm
    ).to(device)

    # Loss-Funktionen pro Task
    ce_motion = nn.CrossEntropyLoss()
    ce_rot = nn.CrossEntropyLoss()
    ce_shape = nn.CrossEntropyLoss()

    # Optimizer: AdamW ist stabil für CNN-Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Output-Ordner pro Run (Zeitstempel)
    run_name = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    best_val_exact = -1.0
    best_path = os.path.join(out_dir, "best.pt")

    print(f"Device: {device}")
    print(f"Train clips: {len(ds_train)} | Val clips: {len(ds_val)} | Test clips: {len(ds_test)}")
    print(f"Output: {out_dir}\n")

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        # Hinweis: Ohne torch.cuda.synchronize() kann dt schwanken,
        # da CUDA-Operationen asynchron sind. Das betrifft nur die Zeitanzeige.
        t0 = time.time()

        train_metrics = train_one_epoch(model, dl_train, device, optimizer, ce_motion, ce_rot, ce_shape)
        val_metrics = evaluate(model, dl_val, device, ce_motion, ce_rot, ce_shape)

        dt = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | {dt:6.1f}s  "
            f"train: loss {train_metrics.loss:.4f} | m {train_metrics.acc_motion:.3f} | r {train_metrics.acc_rot:.3f} | s {train_metrics.acc_shape} | exact {train_metrics.acc_exact:.3f}  "
            f"val: loss {val_metrics.loss:.4f} | m {val_metrics.acc_motion:.3f} | r {val_metrics.acc_rot:.3f} | s {val_metrics.acc_shape}| exact {val_metrics.acc_exact:.3f}"
        )

        # Bestes Modell nach Val-Exact-Match speichern
        if val_metrics.acc_exact > best_val_exact:
            best_val_exact = val_metrics.acc_exact
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics.__dict__,
                    "args": vars(args),
                },
                best_path
            )

    print("\nTraining finished.")
    print(f"Best val exact-match: {best_val_exact:.3f}")
    print(f"Best checkpoint: {best_path}")

    # Test mit bestem Modell
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate(model, dl_test, device, ce_motion, ce_rot, ce_shape)
    print(
        f"\nTEST  loss {test_metrics.loss:.4f} | "
        f"motion {test_metrics.acc_motion:.3f} | rot {test_metrics.acc_rot:.3f} | shape {test_metrics.acc_shape:.3f} | exact {test_metrics.acc_exact:.3f}"
    )

    # Test-Metriken als JSON speichern (für spätere Aggregation über Seeds/Runs)
    results = {
        "seed": int(args.seed),
        "use_tsm": bool(args.use_tsm),
        "n_segment": int(args.n_segment),
        "fold_div": int(args.fold_div),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "test_loss": float(test_metrics.loss),
        "test_acc_motion": float(test_metrics.acc_motion),
        "test_acc_rot": float(test_metrics.acc_rot),
        "test_acc_shape": float(test_metrics.acc_shape),
        "test_acc_exact": float(test_metrics.acc_exact),
    }
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Rotation Confusion Matrix berechnen und in mehreren Formaten abspeichern
    labels_rot = ["ccw", "cw", "none"]  # muss zur ROT_LABELS Reihenfolge passen
    y_true, y_pred = collect_rotation_preds(model, dl_test, device)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # NPY: für spätere Programmauswertung
    np.save(os.path.join(out_dir, "confusion_rot.npy"), cm)

    # CSV: für Tabellen/Excel/BA-Anhang
    cm_csv = os.path.join(out_dir, "confusion_rot.csv")
    with open(cm_csv, "w") as f:
        f.write("true\\pred," + ",".join(labels_rot) + "\n")
        for i, name in enumerate(labels_rot):
            f.write(name + "," + ",".join(str(int(x)) for x in cm[i]) + "\n")

    # PNG: direkt als Abbildung nutzbar
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(3), labels_rot)
    plt.yticks(range(3), labels_rot)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_rot.png"), dpi=200)
    plt.close()

    print(f"Saved test_metrics.json + confusion_rot.(npy/csv/png) to {out_dir}")


if __name__ == "__main__":
    main()
