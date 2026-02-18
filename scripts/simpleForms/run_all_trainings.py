import os
import re
import argparse
import subprocess
from pathlib import Path
from typing import Optional

# Mapping von Dataset-Namen (Teilstring) auf Run-Unterordner
CASE_MAP = {
    "fmotion_frotation": "FIX_MOT_FIX_ROT",
    "fmotion_vrotation": "FIX_MOT_VAR_ROT",
    "vmotion_frotation": "VAR_MOT_FIX_ROT",
    "vmotion_vrotation": "VAR_MOT_VAR_ROT",
}

SEED_RE = re.compile(r"_s(\d+)$")


def infer_case(dataset_name: str) -> Optional[str]:
    """Gibt den Run-Unterordnernamen zurück (z.B. FIX_MOT_VAR_ROT) oder None."""
    for key, case in CASE_MAP.items():
        if key in dataset_name:
            return case
    return None


def infer_seed(dataset_name: str) -> Optional[int]:
    """Extrahiert die Seed-Zahl aus dem Suffix _s123. Gibt None, falls nicht vorhanden."""
    m = SEED_RE.search(dataset_name)
    if not m:
        return None
    return int(m.group(1))


def run_one(
    dataset_path: Path,
    save_dir: Path,
    use_tsm: bool,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    n_segment: int,
    fold_div: int,
    seed: int,
    early_stop: bool,
    patience: int,
    min_delta: float,
    dry_run: bool,
) -> None:
    """
    Startet genau einen Training-Run via subprocess.
    save_dir ist der Ordner, in den train.py seinen Zeitstempel-Run ablegt.
    """
    cmd = [
        "python", "-m", "scripts.simpleForms.train",
        "--data", str(dataset_path),
        "--device", device,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--weight_decay", str(weight_decay),
        "--num_workers", str(num_workers),
        "--n_segment", str(n_segment),
        "--fold_div", str(fold_div),
        "--seed", str(seed),
        "--save_dir", str(save_dir),
    ]

    if use_tsm:
        cmd.append("--use_tsm")

    if early_stop:
        cmd += ["--early_stop", "--patience", str(patience), "--min_delta", str(min_delta)]


    print("\n" + "=" * 80)
    print(("WITH  TSM" if use_tsm else "WITHOUT TSM") + f" | dataset={dataset_path.name} | seed={seed}")
    print("Save dir:", save_dir)
    print("CMD:", " ".join(cmd))
    print("=" * 80)

    if dry_run:
        return

    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/simpleForms",
                    help="Ordner mit den Datensätzen (Unterordner pro dataset)")
    ap.add_argument("--runs_root", type=str, default="runs",
                    help="Oberordner für runs/*")

    # Training-Parameter
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--n_segment", type=int, default=16)
    ap.add_argument("--fold_div", type=int, default=8)

    # Optional: Early Stopping
    ap.add_argument("--early_stop", action="store_true", help="Early stopping aktivieren")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--min_delta", type=float, default=0.001)

    # Verhalten
    ap.add_argument("--dry_run", action="store_true",
                    help="Nur ausgeben, was laufen würde (keine Trainings starten).")
    ap.add_argument("--only_case", type=str, default="",
                    help="Optional: nur einen Case laufen lassen: FIX_MOT_FIX_ROT, ...")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Wenn im Zielordner schon etwas liegt, überspringen (grob).")

    args = ap.parse_args()

    data_root = Path(args.data_root)
    runs_root = Path(args.runs_root)

    if not data_root.exists():
        raise FileNotFoundError(f"data_root nicht gefunden: {data_root}")

    with_root = runs_root / "with_TSM_res"
    without_root = runs_root / "without_TSM_res"
    with_root.mkdir(parents=True, exist_ok=True)
    without_root.mkdir(parents=True, exist_ok=True)

    datasets = sorted([p for p in data_root.iterdir() if p.is_dir()])
    if not datasets:
        print("Keine Dataset-Unterordner gefunden in:", data_root)
        return

    for ds in datasets:
        case = infer_case(ds.name)
        if case is None:
            continue

        if args.only_case and case != args.only_case:
            continue

        seed = infer_seed(ds.name) or 123

        save_dir_with = with_root / case
        save_dir_without = without_root / case
        save_dir_with.mkdir(parents=True, exist_ok=True)
        save_dir_without.mkdir(parents=True, exist_ok=True)

        if args.skip_existing:
            has_with = any(p.is_dir() for p in save_dir_with.iterdir())
            has_without = any(p.is_dir() for p in save_dir_without.iterdir())
            if has_with and has_without:
                print(f"SKIP (existing): {ds.name} -> {case}")
                continue

        run_one(
            dataset_path=ds,
            save_dir=save_dir_without,
            use_tsm=False,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            n_segment=args.n_segment,
            fold_div=args.fold_div,
            seed=seed,
            early_stop=args.early_stop,
            patience=args.patience,
            min_delta=args.min_delta,
            dry_run=args.dry_run
        )

        run_one(
            dataset_path=ds,
            save_dir=save_dir_with,
            use_tsm=True,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            n_segment=args.n_segment,
            fold_div=args.fold_div,
            seed=seed,
            early_stop=args.early_stop,
            patience=args.patience,
            min_delta=args.min_delta,
            dry_run=args.dry_run
        )

    print("\nFertig. Alle geplanten Trainings wurden ausgeführt.")


if __name__ == "__main__":
    main()
