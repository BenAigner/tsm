#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch

# Passe Imports an, falls deine Pfade anders sind:
from models.simpleForms.tsm_cnn import TSM_CNN


CASE_ORDER = ["FIX_MOT_FIX_ROT", "FIX_MOT_VAR_ROT", "VAR_MOT_FIX_ROT", "VAR_MOT_VAR_ROT"]
TSM_ORDER = [False, True]


def load_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def find_first_npz(dataset_root: Path) -> Optional[Path]:
    """
    Greift auf ein echtes Beispiel aus dem Dataset zu, um T,H,W zu bestimmen.
    Erwartete Struktur: dataset_root/train/clips/*.npz
    """
    clip_dir = dataset_root / "train" / "clips"
    if not clip_dir.exists():
        return None
    files = sorted(clip_dir.glob("*.npz"))
    return files[0] if files else None


def infer_input_shape_from_dataset(dataset_root: Path) -> Optional[Tuple[int, int, int, int]]:
    """
    Gibt (T, C, H, W) zurück.
    """
    p = find_first_npz(dataset_root)
    if p is None:
        return None

    data = np.load(p, mmap_mode="r")
    frames = data["frames"]  # [T,H,W,3] uint8
    if frames.ndim != 4 or frames.shape[-1] != 3:
        return None

    T, H, W, C = frames.shape
    return int(T), int(C), int(H), int(W)


def load_ckpt_args(best_pt: Path) -> Optional[Dict[str, Any]]:
    """
    In deinem train.py speicherst du ckpt["args"] mit ab.
    """
    if not best_pt.exists():
        return None
    try:
        ckpt = torch.load(best_pt, map_location="cpu")
        return ckpt.get("args", None)
    except Exception:
        return None


def load_model_from_run(best_pt: Path, device: torch.device) -> Optional[TSM_CNN]:
    """
    Baut das Modell genau wie im Training und lädt state_dict.
    """
    try:
        ckpt = torch.load(best_pt, map_location="cpu")
        args = ckpt.get("args", {})
        n_segment = int(args.get("n_segment", 16))
        fold_div = int(args.get("fold_div", 8))
        use_tsm = bool(args.get("use_tsm", False))

        model = TSM_CNN(
            num_motion_classes=5,
            num_rot_classes=3,
            num_shape_classes=2,
            n_segment=n_segment,
            fold_div=fold_div,
            use_tsm=use_tsm,
        )
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"[WARN] Could not load model from {best_pt}: {e}")
        return None


@torch.no_grad()
def bench_forward_cuda(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int, int],  # (T,C,H,W)
    iters: int,
    warmup: int,
) -> Dict[str, float]:
    """
    Misst reine Forward-Zeit auf CUDA (Batch=1) via CUDA Events.
    Ergebnis in Millisekunden pro Clip.
    """
    assert torch.cuda.is_available(), "CUDA nicht verfügbar, aber du wolltest GPU-only."

    T, C, H, W = input_shape
    x = torch.randn(1, T, C, H, W, device="cuda", dtype=torch.float32)

    # Warmup (wichtig, damit Kernel/CUDNN sich einpendeln)
    for _ in range(warmup):
        _ = model(x)
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    times_ms: List[float] = []

    for _ in range(iters):
        starter.record()
        _ = model(x)
        ender.record()
        torch.cuda.synchronize()
        times_ms.append(float(starter.elapsed_time(ender)))

    arr = np.array(times_ms, dtype=np.float64)

    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "iters": int(iters),
        "warmup": int(warmup),
    }


def collect_run_dirs(runs_root: Path) -> List[Path]:
    """
    Erwartet Struktur:
      runs_root/
        with_TSM_res/<CASE>/<RUN>/
        without_TSM_res/<CASE>/<RUN>/
    """
    out = []
    for variant in ["with_TSM_res", "without_TSM_res"]:
        base = runs_root / variant
        if not base.exists():
            continue
        for case_dir in base.iterdir():
            if not case_dir.is_dir():
                continue
            for run_dir in case_dir.iterdir():
                if run_dir.is_dir():
                    out.append(run_dir)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs_E100_ES", help="Ordner mit with_TSM_res/without_TSM_res")
    ap.add_argument("--out", type=str, default="", help="Output-Ordner (Default: <runs_root>/inference_bench)")
    ap.add_argument("--iters", type=int, default=200, help="Mess-Iterationen (mehr = stabiler)")
    ap.add_argument("--warmup", type=int, default=50, help="Warmup-Iterationen")
    ap.add_argument("--device", type=str, default="cuda", help="nur cuda sinnvoll")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out) if args.out else (runs_root / "inference_bench")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device != "cuda":
        print("[WARN] Du wolltest GPU-only. Setze --device cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA ist nicht verfügbar. Prüfe Treiber/torch installation.")

    # Für Timing oft sinnvoll:
    torch.backends.cudnn.benchmark = True

    run_dirs = collect_run_dirs(runs_root)
    if not run_dirs:
        print("Keine Run-Ordner gefunden. Check runs_root Struktur.")
        return

    device = torch.device("cuda")

    rows: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        best_pt = run_dir / "best.pt"
        metrics_json = run_dir / "test_metrics.json"

        if not best_pt.exists():
            continue
        m = load_json(metrics_json) or {}
        ckpt_args = load_ckpt_args(best_pt) or {}

        # Case/Variante aus Pfad ableiten
        case = run_dir.parent.name
        use_tsm = "with_TSM_res" in str(run_dir)

        seed = m.get("seed", None)
        if seed is None:
            # Notfalls aus Ordnernamen: ..._s456_...
            name = run_dir.name
            sidx = name.find("_s")
            if sidx != -1:
                try:
                    seed = int(name[sidx + 2 : sidx + 5])
                except Exception:
                    seed = None

        data_path = ckpt_args.get("data", "")
        dataset_root = Path(data_path) if data_path else None

        # Input shape bestimmen (T,C,H,W)
        input_shape = None
        if dataset_root and dataset_root.exists():
            input_shape = infer_input_shape_from_dataset(dataset_root)

        if input_shape is None:
            # Fallback: n_segment aus args + Standard 112x112
            T = int(ckpt_args.get("n_segment", 16))
            input_shape = (T, 3, 112, 112)

        model = load_model_from_run(best_pt, device=device)
        if model is None:
            continue

        stats = bench_forward_cuda(model, input_shape, iters=args.iters, warmup=args.warmup)

        row = {
            "case": case,
            "use_tsm": bool(use_tsm),
            "seed": seed,
            "run_dir": str(run_dir),
            "best_pt": str(best_pt),
            "data_root": str(dataset_root) if dataset_root else "",
            "T": input_shape[0],
            "C": input_shape[1],
            "H": input_shape[2],
            "W": input_shape[3],
            **stats,
        }

        # Falls du später Inferenz vs Accuracy korrelieren willst:
        for k in ["test_acc_exact", "test_acc_rot", "test_acc_motion", "test_acc_shape", "test_loss", "epochs_ran", "best_epoch"]:
            if k in m:
                row[k] = m[k]

        rows.append(row)

        print(f"[OK] {case} | {'TSM' if use_tsm else 'noTSM'} | seed={seed} | mean={row['mean_ms']:.3f} ms | p95={row['p95_ms']:.3f} ms")

        # Speicher aufräumen
        del model
        torch.cuda.empty_cache()

    if not rows:
        print("Keine benchbaren Runs gefunden (fehlende best.pt?).")
        return

    df = pd.DataFrame(rows)

    # schöne Sortierung
    df["case"] = pd.Categorical(df["case"], categories=CASE_ORDER, ordered=True)
    df = df.sort_values(["case", "use_tsm", "seed"])

    long_csv = out_dir / "inference_times_long.csv"
    df.to_csv(long_csv, index=False)
    print("Wrote:", long_csv)

    # Summary: mean/std über Seeds pro case×tsm
    summary = df.groupby(["case", "use_tsm"]).agg(
        mean_ms=("mean_ms", "mean"),
        std_ms=("mean_ms", "std"),
        median_ms=("median_ms", "mean"),
        p95_ms=("p95_ms", "mean"),
        mean_epochs_ran=("epochs_ran", "mean") if "epochs_ran" in df.columns else ("mean_ms", "count"),
        n=("mean_ms", "count"),
    ).reset_index()

    summary_csv = out_dir / "inference_times_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print("Wrote:", summary_csv)

    # Konsole: kompakte Tabelle
    print("\n=== Inference summary (ms per clip, batch=1, CUDA forward only) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
