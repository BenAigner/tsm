# ==========================================
# FILE: run_all_trainings_regression.py
# ==========================================
import re
import argparse
import subprocess
from pathlib import Path

CASE_ORDER = ["FIX_MOT_FIX_ROT", "FIX_MOT_VAR_ROT", "VAR_MOT_FIX_ROT", "VAR_MOT_VAR_ROT"]
SEED_RE = re.compile(r"_s(\d+)$")


def run_already_done(save_dir: Path, case: str, seed: int, use_tsm: bool) -> bool:
    """
    True, wenn bereits ein Run für (case, seed, use_tsm) existiert und test_metrics.json hat.
    Erwartete Run-Ordnernamen:
      .../<CASE>/<TIMESTAMP>_s123_tsm/
      .../<CASE>/<TIMESTAMP>_s123_noTSM/
    """
    variant = "with_TSM_res" if use_tsm else "without_TSM_res"
    case_dir = save_dir / variant / case
    if not case_dir.exists():
        return False

    suffix = "_tsm" if use_tsm else "_noTSM"
    needle_seed = f"_s{seed}_"

    for run_dir in case_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if needle_seed in run_dir.name and run_dir.name.endswith(suffix):
            if (run_dir / "test_metrics.json").exists():
                return True

    return False


def infer_case_and_seed(dataset_dirname: str) -> tuple[str, int]:
    """
    Erwartet: <CASE>_s123
    """
    m = SEED_RE.search(dataset_dirname)
    seed = int(m.group(1)) if m else 123
    case = dataset_dirname
    if m:
        case = dataset_dirname[: m.start()]
    return case, seed


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data_root",
        type=str,
        default="data/simpleForms_Regression",
        help="Ordner mit den Dataset-Unterordnern",
    )
    ap.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Base output dir (wird intern in with/without + case sortiert)",
    )

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--n_segment", type=int, default=16)
    ap.add_argument("--fold_div", type=int, default=8)

    ap.add_argument("--normalize_regression", action="store_true")
    ap.add_argument("--lambda_speed", type=float, default=1.0)
    ap.add_argument("--lambda_omega", type=float, default=1.0)

    ap.add_argument("--select_metric", type=str, default="score", choices=["score", "exact", "mae_sum"])
    ap.add_argument("--alpha", type=float, default=0.5)

    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--min_delta", type=float, default=0.001)

    ap.add_argument("--only_case", type=str, default="", help="Optional: nur einen Case laufen lassen")
    ap.add_argument("--dry_run", action="store_true", help="Nur anzeigen, nichts ausführen")

    # AMP
    ap.add_argument("--amp", action="store_true")


    # Optional inference benchmark passt durch
    ap.add_argument("--bench_infer", action="store_true")
    ap.add_argument("--bench_iters", type=int, default=200)
    ap.add_argument("--bench_warmup", type=int, default=50)

    # Skipping already done runs
    ap.add_argument("--skip_done", action="store_true", help="Überspringt bereits fertige Runs")

    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root nicht gefunden: {data_root}")

    datasets = sorted([p for p in data_root.iterdir() if p.is_dir()])
    if not datasets:
        print("Keine Dataset-Unterordner gefunden in:", data_root)
        return

    def sort_key(p: Path):
        case, seed = infer_case_and_seed(p.name)
        ci = CASE_ORDER.index(case) if case in CASE_ORDER else 999
        return (ci, seed)

    datasets = sorted(datasets, key=sort_key)

    for ds in datasets:
        case, seed = infer_case_and_seed(ds.name)
        if args.only_case and case != args.only_case:
            continue

        # ------------------------------------------------------------
        # noTSM
        # ------------------------------------------------------------
        cmd = [
            "python", "-m", "scripts.simpleForms_Regression.train_regression",
            "--data", str(ds),
            "--device", args.device,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--weight_decay", str(args.weight_decay),
            "--num_workers", str(args.num_workers),
            "--n_segment", str(args.n_segment),
            "--fold_div", str(args.fold_div),
            "--seed", str(seed),
            "--save_dir", str(args.save_dir),
            "--lambda_speed", str(args.lambda_speed),
            "--lambda_omega", str(args.lambda_omega),
            "--select_metric", args.select_metric,
            "--alpha", str(args.alpha),
        ]

        if args.normalize_regression:
            cmd.append("--normalize_regression")

        if args.early_stop:
            cmd += ["--early_stop", "--patience", str(args.patience), "--min_delta", str(args.min_delta)]

        if args.amp:
            cmd.append("--amp")


        if args.bench_infer:
            cmd += ["--bench_infer", "--bench_iters", str(args.bench_iters), "--bench_warmup", str(args.bench_warmup)]

        print("\n" + "=" * 90)
        print(f"WITHOUT TSM | case={case} | seed={seed}")
        print("DATA:", ds)

        if args.skip_done and run_already_done(Path(args.save_dir), case, seed, use_tsm=False):
            print(f"[SKIP] noTSM already done for {case} seed {seed}")
        else:
            print("CMD:\n  " + " \\\n  ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)

        print("=" * 90)

        # ------------------------------------------------------------
        # TSM
        # ------------------------------------------------------------
        cmd_tsm = cmd + ["--use_tsm"]

        print("\n" + "=" * 90)
        print(f"WITH TSM | case={case} | seed={seed}")
        print("DATA:", ds)

        if args.skip_done and run_already_done(Path(args.save_dir), case, seed, use_tsm=True):
            print(f"[SKIP] TSM already done for {case} seed {seed}")
        else:
            print("CMD:\n  " + " \\\n  ".join(cmd_tsm))
            if not args.dry_run:
                subprocess.run(cmd_tsm, check=True)

        print("=" * 90)


if __name__ == "__main__":
    main()