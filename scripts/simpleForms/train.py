import os
import json
import time
import csv
import argparse
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from scripts.simpleForms.synth_dataset import SynthMultiLabelNPZ
from models.simpleForms.tsm_cnn import TSM_CNN


# ------------------------------------------------------------
# Kleine Hilfsstruktur für Metriken
# ------------------------------------------------------------
@dataclass
class Metrics:
    loss: float
    acc_motion: float
    acc_rot: float
    acc_shape: float
    acc_exact: float


def set_seed(seed: int) -> None:
    """
    Setzt Seeds möglichst reproduzierbar.
    Achtung: vollständige Deterministik kann Performance kosten.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ce_motion: nn.Module,
    ce_rot: nn.Module,
    ce_shape: nn.Module,
) -> Metrics:
    model.eval()

    total = 0
    total_loss = 0.0
    correct_m = 0
    correct_r = 0
    correct_s = 0
    correct_exact = 0

    for clips, motion, rot, shape in loader:
        clips = clips.to(device, non_blocking=True)
        motion = motion.to(device, non_blocking=True)
        rot = rot.to(device, non_blocking=True)
        shape = shape.to(device, non_blocking=True)

        logits_m, logits_r, logits_s = model(clips)

        loss = (
            ce_motion(logits_m, motion)
            + ce_rot(logits_r, rot)
            + ce_shape(logits_s, shape)
        )

        pred_m = logits_m.argmax(dim=1)
        pred_r = logits_r.argmax(dim=1)
        pred_s = logits_s.argmax(dim=1)

        bsz = clips.size(0)
        total += bsz
        total_loss += float(loss.item()) * bsz

        correct_m += int((pred_m == motion).sum().item())
        correct_r += int((pred_r == rot).sum().item())
        correct_s += int((pred_s == shape).sum().item())
        correct_exact += int(((pred_m == motion) & (pred_r == rot) & (pred_s == shape)).sum().item())

    return Metrics(
        loss=total_loss / max(total, 1),
        acc_motion=correct_m / max(total, 1),
        acc_rot=correct_r / max(total, 1),
        acc_shape=correct_s / max(total, 1),
        acc_exact=correct_exact / max(total, 1),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    ce_motion: nn.Module,
    ce_rot: nn.Module,
    ce_shape: nn.Module,
) -> Metrics:
    model.train()

    total = 0
    total_loss = 0.0
    correct_m = 0
    correct_r = 0
    correct_s = 0
    correct_exact = 0

    for clips, motion, rot, shape in loader:
        clips = clips.to(device, non_blocking=True)
        motion = motion.to(device, non_blocking=True)
        rot = rot.to(device, non_blocking=True)
        shape = shape.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits_m, logits_r, logits_s = model(clips)

        loss = (
            ce_motion(logits_m, motion)
            + ce_rot(logits_r, rot)
            + ce_shape(logits_s, shape)
        )

        loss.backward()
        optimizer.step()

        pred_m = logits_m.argmax(dim=1)
        pred_r = logits_r.argmax(dim=1)
        pred_s = logits_s.argmax(dim=1)

        bsz = clips.size(0)
        total += bsz
        total_loss += float(loss.item()) * bsz

        correct_m += int((pred_m == motion).sum().item())
        correct_r += int((pred_r == rot).sum().item())
        correct_s += int((pred_s == shape).sum().item())
        correct_exact += int(((pred_m == motion) & (pred_r == rot) & (pred_s == shape)).sum().item())

    return Metrics(
        loss=total_loss / max(total, 1),
        acc_motion=correct_m / max(total, 1),
        acc_rot=correct_r / max(total, 1),
        acc_shape=correct_s / max(total, 1),
        acc_exact=correct_exact / max(total, 1),
    )


@torch.no_grad()
def collect_rotation_preds(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: List[int] = []
    ps: List[int] = []
    for clips, _, rot, _ in loader:
        clips = clips.to(device, non_blocking=True)
        rot = rot.to(device, non_blocking=True)
        _, logits_r, _ = model(clips)
        pred_r = logits_r.argmax(dim=1)
        ys.extend(rot.cpu().numpy().tolist())
        ps.extend(pred_r.cpu().numpy().tolist())
    return np.array(ys, dtype=np.int64), np.array(ps, dtype=np.int64)


def main():
    """
    Training-Entry: Argumente parsen, Seeds setzen, DataLoader/Modell initialisieren,
    Training durchführen, bestes Checkpoint speichern, am Ende Test + Confusion Matrix ausgeben.
    Zusätzlich wird eine history.csv geschrieben (Lernkurven).
    """
    ap = argparse.ArgumentParser()

    ap.add_argument("--use_tsm", action="store_true", help="TSM aktivieren (Baseline sonst ohne)")

    ap.add_argument("--data", type=str, required=True,
                    help="Pfad zum Datensatz-Root (enthält train/val/test)")

    # Wir setzen hier bewusst einen größeren Max-Wert und sparen Zeit über Early Stopping.
    ap.add_argument("--epochs", type=int, default=100)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--n_segment", type=int, default=16, help="muss zu T im Datensatz passen")
    ap.add_argument("--fold_div", type=int, default=8)

    ap.add_argument("--device", type=str, default="cuda", help="cuda oder cpu")
    ap.add_argument("--save_dir", type=str, default="runs", help="Output-Verzeichnis für Checkpoints/Logs")

    # Optional: Early Stopping (spart Zeit, ohne die Aussage zu verwässern)
    ap.add_argument("--early_stop", action="store_true", help="Early Stopping aktivieren")
    ap.add_argument("--patience", type=int, default=12, help="wie viele Epochen ohne echte Verbesserung")
    ap.add_argument("--min_delta", type=float, default=0.001, help="Mindestverbesserung auf val_exact")

    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Datasets / Loader
    ds_train = SynthMultiLabelNPZ(args.data, "train")
    ds_val = SynthMultiLabelNPZ(args.data, "val")
    ds_test = SynthMultiLabelNPZ(args.data, "test")

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    model = TSM_CNN(
        num_motion_classes=5,
        num_rot_classes=3,
        num_shape_classes=2,
        n_segment=args.n_segment,
        fold_div=args.fold_div,
        use_tsm=args.use_tsm
    ).to(device)

    ce_motion = nn.CrossEntropyLoss()
    ce_rot = nn.CrossEntropyLoss()
    ce_shape = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Output-Ordner pro Run (Zeitstempel + seed)
    run_name = time.strftime("%Y%m%d-%H%M%S") + f"_s{args.seed}" + ("_tsm" if args.use_tsm else "_noTSM")
    out_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Lernkurven-Logging (CSV)
    history_path = os.path.join(out_dir, "history.csv")
    with open(history_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch", "dt_sec",
            "train_loss", "train_acc_motion", "train_acc_rot", "train_acc_shape", "train_acc_exact",
            "val_loss", "val_acc_motion", "val_acc_rot", "val_acc_shape", "val_acc_exact"
        ])

    best_val_exact = -1.0
    best_epoch = -1
    best_path = os.path.join(out_dir, "best.pt")

    no_improve = 0
    stopped_epoch = 0

    print(f"Device: {device}")
    print(f"Train clips: {len(ds_train)} | Val clips: {len(ds_val)} | Test clips: {len(ds_test)}")
    print(f"Output: {out_dir}")
    print(f"History: {history_path}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, dl_train, device, optimizer, ce_motion, ce_rot, ce_shape)
        val_metrics = evaluate(model, dl_val, device, ce_motion, ce_rot, ce_shape)

        # Hinweis: dt ist nur grob (CUDA ist asynchron), für Benchmarks später separat messen.
        dt = time.time() - t0

        # in history.csv schreiben
        with open(history_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                epoch, round(dt, 4),
                round(train_metrics.loss, 6),
                round(train_metrics.acc_motion, 6),
                round(train_metrics.acc_rot, 6),
                round(train_metrics.acc_shape, 6),
                round(train_metrics.acc_exact, 6),
                round(val_metrics.loss, 6),
                round(val_metrics.acc_motion, 6),
                round(val_metrics.acc_rot, 6),
                round(val_metrics.acc_shape, 6),
                round(val_metrics.acc_exact, 6),
            ])

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | {dt:6.1f}s  "
            f"train: loss {train_metrics.loss:.4f} | m {train_metrics.acc_motion:.3f} | r {train_metrics.acc_rot:.3f} | s {train_metrics.acc_shape:.3f} | exact {train_metrics.acc_exact:.3f}  "
            f"val: loss {val_metrics.loss:.4f} | m {val_metrics.acc_motion:.3f} | r {val_metrics.acc_rot:.3f} | s {val_metrics.acc_shape:.3f} | exact {val_metrics.acc_exact:.3f}"
        )

        # "echte" Verbesserung definieren (min_delta verhindert, dass Mini-Fluktuationen zählen)
        improved = (val_metrics.acc_exact > best_val_exact + args.min_delta)

        if improved:
            best_val_exact = val_metrics.acc_exact
            best_epoch = epoch
            no_improve = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics.__dict__,
                    "args": vars(args),
                },
                best_path
            )
        else:
            no_improve += 1

        if args.early_stop and no_improve >= args.patience:
            print(
                f"\nEarly stopping: keine Verbesserung > min_delta={args.min_delta} "
                f"für {args.patience} Epochen. Stoppe bei Epoche {epoch}."
            )
            stopped_epoch = epoch
            break

    if stopped_epoch == 0:
        stopped_epoch = args.epochs

    print("\nTraining finished.")
    print(f"Best val exact-match: {best_val_exact:.3f} (epoch {best_epoch})")
    print(f"Best checkpoint: {best_path}")

    # Test mit bestem Modell
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate(model, dl_test, device, ce_motion, ce_rot, ce_shape)
    print(
        f"\nTEST  loss {test_metrics.loss:.4f} | "
        f"motion {test_metrics.acc_motion:.3f} | rot {test_metrics.acc_rot:.3f} | shape {test_metrics.acc_shape:.3f} | exact {test_metrics.acc_exact:.3f}"
    )

    # Test-Metriken als JSON speichern (für spätere Aggregation)
    results = {
        "seed": int(args.seed),
        "use_tsm": bool(args.use_tsm),
        "n_segment": int(args.n_segment),
        "fold_div": int(args.fold_div),
        "epochs_max": int(args.epochs),
        "epochs_ran": int(stopped_epoch),
        "early_stop": bool(args.early_stop),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
        "best_val_exact": float(best_val_exact),
        "best_epoch": int(best_epoch),
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

    # Rotation Confusion Matrix
    labels_rot = ["ccw", "cw", "none"]  # muss zu ROT_LABELS Reihenfolge passen
    y_true, y_pred = collect_rotation_preds(model, dl_test, device)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    np.save(os.path.join(out_dir, "confusion_rot.npy"), cm)

    cm_csv = os.path.join(out_dir, "confusion_rot.csv")
    with open(cm_csv, "w") as f:
        f.write("true\\pred," + ",".join(labels_rot) + "\n")
        for i, name in enumerate(labels_rot):
            f.write(name + "," + ",".join(str(int(x)) for x in cm[i]) + "\n")

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

    print(f"\nSaved: history.csv, test_metrics.json, confusion_rot.(npy/csv/png) in {out_dir}")


if __name__ == "__main__":
    main()
