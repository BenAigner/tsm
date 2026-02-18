import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# Reihenfolge der Cases (damit CSVs immer gleich sortiert sind)
CASE_ORDER = ["FIX_MOT_FIX_ROT", "FIX_MOT_VAR_ROT", "VAR_MOT_FIX_ROT", "VAR_MOT_VAR_ROT"]
TSM_ORDER = [False, True]


def load_test_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / "test_metrics.json"
    if not p.exists():
        return None
    try:
        with open(p, "r") as f:
            d = json.load(f)
        d["_run_dir"] = str(run_dir)
        return d
    except Exception:
        return None


def load_history(run_dir: Path) -> Optional[pd.DataFrame]:
    p = run_dir / "history.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df
    except Exception:
        return None


def collect_runs(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not root.exists():
        return pd.DataFrame()

    for case_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        case_name = case_dir.name
        for run_dir in sorted([p for p in case_dir.iterdir() if p.is_dir()]):
            d = load_test_metrics(run_dir)
            if d is None:
                continue
            d["_case"] = case_name
            rows.append(d)

    return pd.DataFrame(rows)


def fmt_mean_std(mean: float, std: float, decimals: int = 3) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def make_overview_table(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    g = df.groupby(["_case", "use_tsm"])[metric_cols].agg(["mean", "std"])
    g = g.reset_index()

    g["_case"] = pd.Categorical(g["_case"], categories=CASE_ORDER, ordered=True)
    g["use_tsm"] = pd.Categorical(g["use_tsm"], categories=TSM_ORDER, ordered=True)
    g = g.sort_values(["_case", "use_tsm"])

    rows = []
    for case in CASE_ORDER:
        row = {"case": case}
        sub = g[g["_case"] == case]
        if sub.empty:
            rows.append(row)
            continue

        for use_tsm in TSM_ORDER:
            sub2 = sub[sub["use_tsm"] == use_tsm]
            tag = "TSM" if use_tsm else "noTSM"
            if sub2.empty:
                for m in metric_cols:
                    row[f"{m}_{tag}"] = ""
                continue

            sub2 = sub2.iloc[0]
            for m in metric_cols:
                mean = float(sub2[(m, "mean")]) if (m, "mean") in sub2.index else float("nan")
                std = float(sub2[(m, "std")]) if (m, "std") in sub2.index else float("nan")
                dec = 4 if "loss" in m else 3
                row[f"{m}_{tag}"] = fmt_mean_std(mean, std, decimals=dec)

        rows.append(row)

    out = pd.DataFrame(rows)

    col_order = ["case"]
    for m in metric_cols:
        col_order += [f"{m}_noTSM", f"{m}_TSM"]
    out = out[col_order]
    return out


def make_gain_table(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    g = df.groupby(["_case", "use_tsm"])[metric_cols].mean().reset_index()

    g["_case"] = pd.Categorical(g["_case"], categories=CASE_ORDER, ordered=True)
    g = g.sort_values(["_case", "use_tsm"])

    rows = []
    for case in CASE_ORDER:
        row = {"case": case}
        sub = g[g["_case"] == case]
        if sub.empty:
            rows.append(row)
            continue

        has_no = (sub["use_tsm"] == False).any()
        has_tsm = (sub["use_tsm"] == True).any()
        if not (has_no and has_tsm):
            rows.append(row)
            continue

        no = sub[sub["use_tsm"] == False].iloc[0]
        tsm = sub[sub["use_tsm"] == True].iloc[0]

        for m in metric_cols:
            mn = float(no[m])
            mt = float(tsm[m])
            gain = mt - mn
            dec = 4 if "loss" in m else 3
            row[f"{m}_mean_noTSM"] = round(mn, dec)
            row[f"{m}_mean_TSM"] = round(mt, dec)
            row[f"{m}_gain"] = round(gain, dec)

        rows.append(row)

    out = pd.DataFrame(rows)

    col_order = ["case"]
    for m in metric_cols:
        col_order += [f"{m}_mean_noTSM", f"{m}_mean_TSM", f"{m}_gain"]
    out = out[col_order]
    return out


def _mean_std_by_case(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = df.groupby(["_case", "use_tsm"])[metric].agg(["mean", "std"]).reset_index()
    g["_case"] = pd.Categorical(g["_case"], categories=CASE_ORDER, ordered=True)
    g["use_tsm"] = pd.Categorical(g["use_tsm"], categories=TSM_ORDER, ordered=True)
    g = g.sort_values(["_case", "use_tsm"])
    return g


def plot_bars_mean_std(df_long: pd.DataFrame, metric: str, out_png: Path, title: str) -> None:
    g = _mean_std_by_case(df_long, metric)

    cases = CASE_ORDER
    x = list(range(len(cases)))
    width = 0.35

    means_no = []
    stds_no = []
    means_tsm = []
    stds_tsm = []

    for c in cases:
        sub = g[g["_case"] == c]
        no = sub[sub["use_tsm"] == False]
        tsm = sub[sub["use_tsm"] == True]

        means_no.append(float(no["mean"].iloc[0]) if not no.empty else float("nan"))
        stds_no.append(float(no["std"].iloc[0]) if not no.empty else 0.0)

        means_tsm.append(float(tsm["mean"].iloc[0]) if not tsm.empty else float("nan"))
        stds_tsm.append(float(tsm["std"].iloc[0]) if not tsm.empty else 0.0)

    plt.figure(figsize=(9, 4.5))
    plt.bar([i - width/2 for i in x], means_no, width, yerr=stds_no, capsize=4, label="noTSM")
    plt.bar([i + width/2 for i in x], means_tsm, width, yerr=stds_tsm, capsize=4, label="TSM")
    plt.xticks(x, cases, rotation=15, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_gain(df_long: pd.DataFrame, metric: str, out_png: Path, title: str) -> None:
    # Gain = mean(TSM) - mean(noTSM)
    g = df_long.groupby(["_case", "use_tsm"])[metric].mean().reset_index()
    g["_case"] = pd.Categorical(g["_case"], categories=CASE_ORDER, ordered=True)
    g = g.sort_values(["_case", "use_tsm"])

    gains = []
    for c in CASE_ORDER:
        sub = g[g["_case"] == c]
        no = sub[sub["use_tsm"] == False]
        tsm = sub[sub["use_tsm"] == True]
        if no.empty or tsm.empty:
            gains.append(float("nan"))
        else:
            gains.append(float(tsm[metric].iloc[0] - no[metric].iloc[0]))

    plt.figure(figsize=(9, 4.5))
    plt.bar(list(range(len(CASE_ORDER))), gains)
    plt.xticks(list(range(len(CASE_ORDER))), CASE_ORDER, rotation=15, ha="right")
    plt.axhline(0.0, linewidth=1)
    plt.ylabel(f"gain ({metric})")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def pick_representative_pair(df_long: pd.DataFrame, case: str) -> Optional[Tuple[int, str, str]]:
    """
    Wählt für einen Case einen Seed, bei dem noTSM am schlechtesten war,
    und sucht dann die passende TSM-Run-Dir für denselben Seed.
    Rückgabe: (seed, run_dir_no, run_dir_tsm)
    """
    sub = df_long[df_long["_case"] == case]
    if sub.empty:
        return None

    # bevorzugt anhand test_acc_exact; wenn nicht da, nimm rot
    metric = "test_acc_exact" if "test_acc_exact" in sub.columns else "test_acc_rot"
    no = sub[sub["use_tsm"] == False]
    tsm = sub[sub["use_tsm"] == True]
    if no.empty or tsm.empty:
        return None

    # seed mit schlechtestem noTSM
    worst = no.sort_values(metric, ascending=True).iloc[0]
    seed = int(worst["seed"])
    run_no = str(worst["_run_dir"])

    tsm_same_seed = tsm[tsm["seed"] == seed]
    if tsm_same_seed.empty:
        return None
    run_tsm = str(tsm_same_seed.iloc[0]["_run_dir"])
    return seed, run_no, run_tsm


def plot_learning_curve_pair(run_no: Path, run_tsm: Path, out_png: Path, title: str) -> None:
    h_no = load_history(run_no)
    h_tsm = load_history(run_tsm)
    if h_no is None or h_tsm is None:
        return

    # val_exact ist das, was du optimierst
    if "val_acc_exact" not in h_no.columns or "val_acc_exact" not in h_tsm.columns:
        return

    plt.figure(figsize=(9, 4.5))
    plt.plot(h_no["epoch"], h_no["val_acc_exact"], label="noTSM val_exact")
    plt.plot(h_tsm["epoch"], h_tsm["val_acc_exact"], label="TSM val_exact")
    plt.xlabel("epoch")
    plt.ylabel("val_acc_exact")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--with_dir", type=str, default="with_TSM_res")
    ap.add_argument("--without_dir", type=str, default="without_TSM_res")
    ap.add_argument("--out", type=str, default="runs/summary_pretty")

    ap.add_argument("--make_plots", action="store_true", help="Erzeuge PNG-Plots in out/")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    with_root = runs_root / args.with_dir
    without_root = runs_root / args.without_dir

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_with = collect_runs(with_root)
    df_without = collect_runs(without_root)

    df = pd.concat([df_with, df_without], ignore_index=True)
    if df.empty:
        print("Keine test_metrics.json gefunden. Check deine runs/* Struktur.")
        return

    # use_tsm normalisieren
    if "use_tsm" in df.columns:
        df["use_tsm"] = df["use_tsm"].astype(bool)
    else:
        df["use_tsm"] = df["_run_dir"].astype(str).str.contains("with_TSM_res")

    # gewünschte Metriken (plus training meta falls vorhanden)
    metric_cols = [
        "test_loss",
        "test_acc_motion",
        "test_acc_rot",
        "test_acc_shape",
        "test_acc_exact",
        # optional: training meta (falls vorhanden)
        "epochs_ran",
        "best_epoch",
        "best_val_exact",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    # “Long” Tabelle: alle Runs, sortiert
    df_long = df.copy()
    df_long["_case"] = pd.Categorical(df_long["_case"], categories=CASE_ORDER, ordered=True)
    # seed kann fehlen -> robust sortieren
    sort_cols = ["_case", "use_tsm"]
    if "seed" in df_long.columns:
        sort_cols.append("seed")
    df_long = df_long.sort_values(sort_cols)

    long_csv = out_dir / "all_runs_long.csv"
    df_long.to_csv(long_csv, index=False)
    print("Wrote:", long_csv)

    # Übersicht (mean ± std) nur für "klassische" metrics (keine epochs)
    classic_metrics = [c for c in metric_cols if c.startswith("test_")]
    overview = make_overview_table(df_long, metric_cols=classic_metrics)
    overview_csv = out_dir / "overview_mean_std.csv"
    overview.to_csv(overview_csv, index=False)
    print("Wrote:", overview_csv)

    # Gains (nur test metrics)
    gain = make_gain_table(df_long, metric_cols=classic_metrics)
    gain_csv = out_dir / "tsm_gain.csv"
    gain.to_csv(gain_csv, index=False)
    print("Wrote:", gain_csv)

    # Optional: zusätzliche Übersicht für epochs/best_epoch/best_val_exact
    extra = [c for c in ["epochs_ran", "best_epoch", "best_val_exact"] if c in df_long.columns]
    if extra:
        extra_overview = make_overview_table(df_long, metric_cols=extra)
        extra_csv = out_dir / "training_meta_mean_std.csv"
        extra_overview.to_csv(extra_csv, index=False)
        print("Wrote:", extra_csv)

    # Plots
    if args.make_plots:
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        if "test_acc_exact" in df_long.columns:
            plot_bars_mean_std(
                df_long, "test_acc_exact",
                plots_dir / "bar_exact_mean_std.png",
                "Exact accuracy: noTSM vs TSM (mean ± std over seeds)"
            )
            plot_gain(
                df_long, "test_acc_exact",
                plots_dir / "gain_exact.png",
                "Gain (TSM - noTSM) for Exact accuracy"
            )

        if "test_acc_rot" in df_long.columns:
            plot_bars_mean_std(
                df_long, "test_acc_rot",
                plots_dir / "bar_rot_mean_std.png",
                "Rotation accuracy: noTSM vs TSM (mean ± std over seeds)"
            )
            plot_gain(
                df_long, "test_acc_rot",
                plots_dir / "gain_rot.png",
                "Gain (TSM - noTSM) for Rotation accuracy"
            )

        if "epochs_ran" in df_long.columns:
            plot_bars_mean_std(
                df_long, "epochs_ran",
                plots_dir / "bar_epochs_ran_mean_std.png",
                "Epochs actually run (early stopping): noTSM vs TSM"
            )

        # Learning curves: pro Case ein repräsentativer Seed (worst noTSM) wenn möglich
        for case in CASE_ORDER:
            pair = pick_representative_pair(df_long, case)
            if pair is None:
                continue
            seed, run_no, run_tsm = pair
            out_png = plots_dir / f"curve_val_exact_{case}_s{seed}.png"
            plot_learning_curve_pair(
                Path(run_no), Path(run_tsm),
                out_png,
                f"Validation exact over epochs ({case}, seed {seed})"
            )

        print("Wrote plots to:", plots_dir)

    print("\nDone. Outputs in:", out_dir)


if __name__ == "__main__":
    main()
