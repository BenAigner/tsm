#!/usr/bin/env python3
"""
summarize_runs_regression.py

Liest alle test_metrics.json aus:
  <runs_root>/
    with_TSM_res/<CASE>/<RUN>/test_metrics.json
    without_TSM_res/<CASE>/<RUN>/test_metrics.json

Erzeugt:
  - all_runs_long.csv
  - overview_mean_std.csv                (mean ± std: noTSM vs TSM)
  - gain_table.csv                       (mean-basierter Gain)
  - gain_table_paired.csv                (paired Gain pro Seed, statistisch sauberer)
  - barplots/*.png                       (mean±std Balken, pro Metrik)
  - boxplots/*.png                       (Verteilung über Seeds, pro Metrik)
  - gainplots/*.png                      (paired gain mean±std pro Case)
  - tradeoff_acc_vs_infer.png            (optional, wenn inference_ms_per_forward vorhanden)

Hinweis zu Gains:
  - Accuracy: größer ist besser -> gain = (TSM - noTSM)
  - Fehlermaße (MAE/RMSE/Loss): kleiner ist besser -> gain = (noTSM - TSM)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CASE_ORDER = ["FIX_MOT_FIX_ROT", "FIX_MOT_VAR_ROT", "VAR_MOT_FIX_ROT", "VAR_MOT_VAR_ROT"]
TSM_ORDER = [False, True]


# -----------------------
# IO helpers
# -----------------------
def load_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def collect_runs(runs_root: Path) -> pd.DataFrame:
    """
    Sammelt alle test_metrics.json aus with_TSM_res und without_TSM_res.
    Erwartete Struktur:
      runs_root/
        with_TSM_res/<CASE>/<RUN>/test_metrics.json
        without_TSM_res/<CASE>/<RUN>/test_metrics.json
    """
    rows: List[Dict[str, Any]] = []

    for variant in ["with_TSM_res", "without_TSM_res"]:
        base = runs_root / variant
        if not base.exists():
            continue

        for case_dir in base.iterdir():
            if not case_dir.is_dir():
                continue

            case = case_dir.name

            for run_dir in case_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                p = run_dir / "test_metrics.json"
                d = load_json(p)
                if not d:
                    continue

                d["_run_dir"] = str(run_dir)
                d["_case"] = case
                d["use_tsm"] = True if variant == "with_TSM_res" else False
                rows.append(d)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def fmt_mean_std(mean: float, std: float, decimals: int = 3) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _to_bool_filter(x: str) -> Optional[bool]:
    if x == "":
        return None
    if x in ("1", "true", "True", "yes", "Yes"):
        return True
    if x in ("0", "false", "False", "no", "No"):
        return False
    raise ValueError(f"Invalid bool filter: {x}")


def pick_metric_columns(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
    """
    Wählt sinnvolle Metrik-Spalten aus.
    Für speed/omega bevorzugen wir denorm, falls vorhanden.
    Rückgabe:
      metric_cols: Liste der Spalten im df
      pretty_names: Mapping für Plot-Beschriftung
    """
    metric_cols: List[str] = []
    pretty: Dict[str, str] = {}

    # Accuracy
    if "test_acc_exact" in df.columns:
        metric_cols.append("test_acc_exact")
        pretty["test_acc_exact"] = "Exact Acc"

    # Optional weitere Accs (falls du sie im train_regression speicherst)
    if "test_acc_motion" in df.columns:
        metric_cols.append("test_acc_motion")
        pretty["test_acc_motion"] = "Acc Motion"
    if "test_acc_rot" in df.columns:
        metric_cols.append("test_acc_rot")
        pretty["test_acc_rot"] = "Acc Rot"
    if "test_acc_shape" in df.columns:
        metric_cols.append("test_acc_shape")
        pretty["test_acc_shape"] = "Acc Shape"

    # Speed MAE: prefer denorm
    if "test_mae_speed_denorm" in df.columns:
        metric_cols.append("test_mae_speed_denorm")
        pretty["test_mae_speed_denorm"] = "MAE Speed (px/frame)"
    elif "test_mae_speed" in df.columns:
        metric_cols.append("test_mae_speed")
        pretty["test_mae_speed"] = "MAE Speed (norm)"

    # Omega MAE: prefer denorm
    if "test_mae_omega_denorm" in df.columns:
        metric_cols.append("test_mae_omega_denorm")
        pretty["test_mae_omega_denorm"] = "MAE Omega (deg/frame)"
    elif "test_mae_omega" in df.columns:
        metric_cols.append("test_mae_omega")
        pretty["test_mae_omega"] = "MAE Omega (norm)"

    # RMSE (optional)
    if "test_rmse_speed_denorm" in df.columns:
        metric_cols.append("test_rmse_speed_denorm")
        pretty["test_rmse_speed_denorm"] = "RMSE Speed (px/frame)"
    elif "test_rmse_speed" in df.columns:
        metric_cols.append("test_rmse_speed")
        pretty["test_rmse_speed"] = "RMSE Speed (norm)"

    if "test_rmse_omega_denorm" in df.columns:
        metric_cols.append("test_rmse_omega_denorm")
        pretty["test_rmse_omega_denorm"] = "RMSE Omega (deg/frame)"
    elif "test_rmse_omega" in df.columns:
        metric_cols.append("test_rmse_omega")
        pretty["test_rmse_omega"] = "RMSE Omega (norm)"

    # Inference (optional)
    if "inference_ms_per_forward" in df.columns:
        metric_cols.append("inference_ms_per_forward")
        pretty["inference_ms_per_forward"] = "Inference (ms/forward)"

    return metric_cols, pretty


def make_overview_mean_std(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """
    1 Zeile pro CASE.
    Spalten: pro Metrik jeweils noTSM und TSM als "mean ± std"
    """
    g = df.groupby(["_case", "use_tsm"])[metric_cols].agg(["mean", "std"]).reset_index()

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
                dec = 4 if ("loss" in m) else (3 if "acc" in m else 4)
                row[f"{m}_{tag}"] = fmt_mean_std(mean, std, decimals=dec)

        rows.append(row)

    out = pd.DataFrame(rows)

    col_order = ["case"]
    for m in metric_cols:
        col_order += [f"{m}_noTSM", f"{m}_TSM"]
    return out[col_order]


def is_error_metric(metric_name: str) -> bool:
    """
    True wenn kleiner besser ist.
    """
    name = metric_name.lower()
    return ("mae" in name) or ("rmse" in name) or ("loss" in name)


def make_gain_table(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """
    Mean-basierte Gains:
      - Accuracy: gain = mean(TSM) - mean(noTSM)
      - Error:    gain = mean(noTSM) - mean(TSM)
    """
    g = df.groupby(["_case", "use_tsm"])[metric_cols].mean().reset_index()

    g["_case"] = pd.Categorical(g["_case"], categories=CASE_ORDER, ordered=True)
    g = g.sort_values(["_case", "use_tsm"])

    rows: List[Dict[str, Any]] = []
    for case in CASE_ORDER:
        row: Dict[str, Any] = {"case": case}
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
            if m not in sub.columns:
                continue
            mn = float(no[m])
            mt = float(tsm[m])

            gain = (mn - mt) if is_error_metric(m) else (mt - mn)

            dec = 4 if ("loss" in m) else (3 if "acc" in m else 4)
            row[f"{m}_mean_noTSM"] = round(mn, dec)
            row[f"{m}_mean_TSM"] = round(mt, dec)
            row[f"{m}_gain"] = round(gain, dec)

        rows.append(row)

    out = pd.DataFrame(rows)

    col_order = ["case"]
    for m in metric_cols:
        col_order += [f"{m}_mean_noTSM", f"{m}_mean_TSM", f"{m}_gain"]
    # Filter only existing columns (wenn z.B. inference nicht in beiden Varianten existiert)
    col_order = [c for c in col_order if c in out.columns]
    return out[col_order]


def make_paired_gain_table(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """
    Paired Gain pro CASE über Seeds:
      gain(seed) = TSM(seed) - noTSM(seed)  (Accuracy)
      gain(seed) = noTSM(seed) - TSM(seed)  (Error metrics)
    Gibt mean/std/n der paired gains aus.
    """
    if "seed" not in df.columns:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for case in CASE_ORDER:
        dcase = df[df["_case"] == case].copy()
        if dcase.empty:
            continue

        # Pivot: index=seed, columns=use_tsm, values=metric_cols
        piv = dcase.pivot_table(index="seed", columns="use_tsm", values=metric_cols, aggfunc="mean")

        # Erwartet bool columns False/True
        if (False not in piv.columns.levels[1]) or (True not in piv.columns.levels[1]):
            continue

        row: Dict[str, Any] = {"case": case}
        for m in metric_cols:
            try:
                no = piv[(m, False)]
                ts = piv[(m, True)]
            except Exception:
                continue

            paired = pd.concat([no, ts], axis=1, keys=["no", "ts"]).dropna()
            if paired.empty:
                continue

            gain = (paired["no"] - paired["ts"]) if is_error_metric(m) else (paired["ts"] - paired["no"])

            row[f"{m}_gain_mean"] = float(gain.mean())
            row[f"{m}_gain_std"] = float(gain.std(ddof=1)) if len(gain) > 1 else 0.0
            row[f"{m}_gain_n"] = int(len(gain))

        rows.append(row)

    out = pd.DataFrame(rows)

    # Ordnung der Spalten
    col_order = ["case"]
    for m in metric_cols:
        col_order += [f"{m}_gain_mean", f"{m}_gain_std", f"{m}_gain_n"]
    col_order = [c for c in col_order if c in out.columns]
    return out[col_order]


# -----------------------
# Plotting
# -----------------------
def plot_bar_mean_std(summary: pd.DataFrame, pretty_name: str, out_png: Path) -> None:
    """
    Barplot pro CASE, zwei Balken (noTSM, TSM) mit std Errorbars.
    summary erwartet Spalten: case, use_tsm, mean, std
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    s = summary.copy()
    s["case"] = pd.Categorical(s["case"], categories=CASE_ORDER, ordered=True)
    s["use_tsm"] = pd.Categorical(s["use_tsm"], categories=TSM_ORDER, ordered=True)
    s = s.sort_values(["case", "use_tsm"])

    cases = CASE_ORDER
    x = np.arange(len(cases))
    width = 0.35

    no = s[s["use_tsm"] == False].set_index("case")
    ts = s[s["use_tsm"] == True].set_index("case")

    mean_no = [float(no.loc[c, "mean"]) if c in no.index else np.nan for c in cases]
    std_no  = [float(no.loc[c, "std"])  if c in no.index else np.nan for c in cases]
    mean_ts = [float(ts.loc[c, "mean"]) if c in ts.index else np.nan for c in cases]
    std_ts  = [float(ts.loc[c, "std"])  if c in ts.index else np.nan for c in cases]

    plt.figure(figsize=(9, 4.5))
    plt.bar(x - width/2, mean_no, width, yerr=std_no, capsize=4, label="noTSM")
    plt.bar(x + width/2, mean_ts, width, yerr=std_ts, capsize=4, label="TSM")
    plt.xticks(x, cases, rotation=15, ha="right")
    plt.ylabel(pretty_name)
    plt.title(f"{pretty_name} (mean ± std over seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_box_by_case(df: pd.DataFrame, metric: str, pretty_name: str, out_png: Path) -> None:
    """
    Boxplot: pro Case je 2 Boxen (noTSM/TSM), über Seeds.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    dfp = df.copy()
    dfp = dfp[dfp["_case"].isin(CASE_ORDER)]
    dfp["_case"] = pd.Categorical(dfp["_case"], categories=CASE_ORDER, ordered=True)

    plt.figure(figsize=(11, 5.0))

    positions = []
    data = []
    labels = []
    pos = 0.0

    for case in CASE_ORDER:
        for use_tsm in [False, True]:
            sub = dfp[(dfp["_case"] == case) & (dfp["use_tsm"].astype(bool) == use_tsm)]
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna().values
            data.append(vals)
            positions.append(pos)
            labels.append(f"{case}\n{'TSM' if use_tsm else 'noTSM'}")
            pos += 1.0
        pos += 0.6  # gap between cases

    plt.boxplot(data, positions=positions, widths=0.65, showfliers=True)
    plt.xticks(positions, labels, rotation=0)
    plt.ylabel(pretty_name)
    plt.title(f"{pretty_name} distribution over seeds (per case)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_gain_paired(paired_gain: pd.DataFrame, metric: str, pretty_name: str, out_png: Path) -> None:
    """
    Balkenplot: paired gain mean ± std pro Case.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if paired_gain.empty:
        return
    if f"{metric}_gain_mean" not in paired_gain.columns:
        return

    pg = paired_gain.copy()
    pg["case"] = pd.Categorical(pg["case"], categories=CASE_ORDER, ordered=True)
    pg = pg.sort_values("case")

    cases = CASE_ORDER
    x = np.arange(len(cases))
    mean = [float(pg.loc[pg["case"] == c, f"{metric}_gain_mean"].values[0]) if (pg["case"] == c).any() else np.nan for c in cases]
    std  = [float(pg.loc[pg["case"] == c, f"{metric}_gain_std"].values[0]) if (pg["case"] == c).any() else np.nan for c in cases]
    n    = [int(pg.loc[pg["case"] == c, f"{metric}_gain_n"].values[0]) if (pg["case"] == c).any() else 0 for c in cases]

    plt.figure(figsize=(9, 4.5))
    plt.bar(x, mean, yerr=std, capsize=4)
    plt.xticks(x, cases, rotation=15, ha="right")
    plt.ylabel(f"Gain ({pretty_name})")
    plt.title(f"Paired Gain (mean ± std over paired seeds) | n={n}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_tradeoff(df: pd.DataFrame, out_png: Path) -> None:
    """
    Scatter: Accuracy vs Inference time.
    """
    if "inference_ms_per_forward" not in df.columns or "test_acc_exact" not in df.columns:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)

    x = pd.to_numeric(df["inference_ms_per_forward"], errors="coerce")
    y = pd.to_numeric(df["test_acc_exact"], errors="coerce")
    m = x.notna() & y.notna()
    dfp = df[m].copy()
    if dfp.empty:
        return

    plt.figure(figsize=(6.2, 4.8))
    for use_tsm in [False, True]:
        sub = dfp[dfp["use_tsm"].astype(bool) == use_tsm]
        plt.scatter(
            sub["inference_ms_per_forward"],
            sub["test_acc_exact"],
            label=("TSM" if use_tsm else "noTSM"),
            s=30,
            alpha=0.75,
        )
    plt.xlabel("Inference time (ms/forward)")
    plt.ylabel("Test exact accuracy")
    plt.title("Accuracy vs inference time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs_root",
        type=str,
        default="runs/ResShift/Regression",
        help="Root mit with_TSM_res/without_TSM_res",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Output dir (default: <runs_root>/summary_regression)",
    )

    # Filter (damit Serien nicht vermischt werden)
    ap.add_argument("--filter_freeze_bn", type=str, default="",
                    help="''=egal, '1'=nur freeze_bn=True, '0'=nur freeze_bn=False")
    ap.add_argument("--filter_amp", type=str, default="",
                    help="''=egal, '1'=nur amp=True, '0'=nur amp=False")
    ap.add_argument("--filter_normalize_regression", type=str, default="",
                    help="''=egal, '1'=nur normalize_regression=True, '0'=nur False")

    # Plot toggles
    ap.add_argument("--no_boxplots", action="store_true", help="Boxplots nicht erzeugen")
    ap.add_argument("--no_gainplots", action="store_true", help="Paired gain plots nicht erzeugen")
    ap.add_argument("--no_tradeoff", action="store_true", help="Tradeoff-Plot nicht erzeugen")

    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out) if args.out else (runs_root / "summary_regression")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_runs(runs_root)
    if df.empty:
        print("Keine test_metrics.json gefunden. Check runs_root Struktur.")
        return

    # Typen / Ordnung
    df["_case"] = pd.Categorical(df["_case"], categories=CASE_ORDER, ordered=True)
    df["use_tsm"] = df["use_tsm"].astype(bool)

    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")

    # Filter anwenden (nur wenn Spalten existieren)
    f_freeze = _to_bool_filter(args.filter_freeze_bn)
    f_amp = _to_bool_filter(args.filter_amp)
    f_norm = _to_bool_filter(args.filter_normalize_regression)

    if f_freeze is not None and "freeze_bn" in df.columns:
        df = df[df["freeze_bn"].astype(bool) == f_freeze]
    if f_amp is not None and "amp" in df.columns:
        df = df[df["amp"].astype(bool) == f_amp]
    if f_norm is not None and "normalize_regression" in df.columns:
        df = df[df["normalize_regression"].astype(bool) == f_norm]

    if df.empty:
        print("Nach Filtern keine Runs übrig. Prüfe --filter_* und runs_root.")
        return

    df = df.sort_values(["_case", "use_tsm", "seed"] if "seed" in df.columns else ["_case", "use_tsm"])

    # Long CSV
    long_csv = out_dir / "all_runs_long.csv"
    df.to_csv(long_csv, index=False)
    print("Wrote:", long_csv)

    # Metriken auswählen
    metric_cols, pretty = pick_metric_columns(df)

    # Overview mean±std
    overview = make_overview_mean_std(df, metric_cols)
    overview_csv = out_dir / "overview_mean_std.csv"
    overview.to_csv(overview_csv, index=False)
    print("Wrote:", overview_csv)

    # Gain (mean-basiert)
    gain = make_gain_table(df, metric_cols)
    gain_csv = out_dir / "gain_table.csv"
    gain.to_csv(gain_csv, index=False)
    print("Wrote:", gain_csv)

    # Gain (paired pro seed)
    paired_gain = make_paired_gain_table(df, metric_cols)
    paired_gain_csv = out_dir / "gain_table_paired.csv"
    paired_gain.to_csv(paired_gain_csv, index=False)
    print("Wrote:", paired_gain_csv)

    # Barplots (mean±std)
    plot_dir = out_dir / "barplots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for m in metric_cols:
        # Inferenzzeit nicht als noTSM/TSM Barplot pro Case, wenn nicht in beiden vorhanden
        # (geht trotzdem, aber ist oft unvollständig)
        if m == "inference_ms_per_forward":
            continue

        # Nur numerische Metriken plotten
        if m not in df.columns:
            continue

        s = df.groupby(["_case", "use_tsm"])[m].agg(["mean", "std"]).reset_index()
        s = s.rename(columns={"_case": "case"})
        s["mean"] = pd.to_numeric(s["mean"], errors="coerce")
        s["std"] = pd.to_numeric(s["std"], errors="coerce")

        out_png = plot_dir / f"bar_{m}.png"
        plot_bar_mean_std(s, pretty_name=pretty.get(m, m), out_png=out_png)
        print("Wrote:", out_png)

    # Boxplots
    if not args.no_boxplots:
        box_dir = out_dir / "boxplots"
        box_dir.mkdir(parents=True, exist_ok=True)

        for m in metric_cols:
            if m not in df.columns:
                continue
            if m == "inference_ms_per_forward":
                continue
            out_png = box_dir / f"box_{m}.png"
            plot_box_by_case(df, m, pretty.get(m, m), out_png)
            print("Wrote:", out_png)

    # Paired gain plots
    if not args.no_gainplots and not paired_gain.empty:
        gp_dir = out_dir / "gainplots"
        gp_dir.mkdir(parents=True, exist_ok=True)

        for m in metric_cols:
            if m == "inference_ms_per_forward":
                continue
            out_png = gp_dir / f"gain_{m}.png"
            plot_gain_paired(paired_gain, m, pretty.get(m, m), out_png)
            if out_png.exists():
                print("Wrote:", out_png)

    # Tradeoff: acc vs inference
    if not args.no_tradeoff:
        plot_tradeoff(df, out_dir / "tradeoff_acc_vs_infer.png")
        if (out_dir / "tradeoff_acc_vs_infer.png").exists():
            print("Wrote:", out_dir / "tradeoff_acc_vs_infer.png")

    # Console preview
    print("\n=== Summary (overview_mean_std.csv) ===")
    print(overview.to_string(index=False))

    print("\nDone. Outputs in:", out_dir)


if __name__ == "__main__":
    main()