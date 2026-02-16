import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd


def load_test_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Erwartet: run_dir/test_metrics.json
    Gibt dict zurück oder None, falls nicht vorhanden/kaputt.
    """
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


def collect_runs(root: Path) -> pd.DataFrame:
    """
    Läuft über:
      root/<CASE>/<RUN_TIMESTAMP...>/test_metrics.json
    und sammelt alles in ein DataFrame.
    """
    rows: List[Dict[str, Any]] = []

    if not root.exists():
        return pd.DataFrame()

    for case_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        case_name = case_dir.name

        # run subdirs
        for run_dir in sorted([p for p in case_dir.iterdir() if p.is_dir()]):
            d = load_test_metrics(run_dir)
            if d is None:
                continue
            d["_case"] = case_name
            rows.append(d)

    return pd.DataFrame(rows)


def mean_std_table(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    """
    Aggregiert mean/std über Gruppen.
    """
    agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std"])
    # flache Spaltennamen
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    return agg.reset_index()


def compute_tsm_gain(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """
    Berechnet pro CASE den Unterschied (TSM - noTSM) auf den MEAN-Werten über Seeds.
    Dazu:
      1) mean über seeds separat für use_tsm True/False
      2) pivot
      3) diff
    """
    # Mittelwerte über Seeds je case+use_tsm
    g = df.groupby(["_case", "use_tsm"])[metric_cols].mean().reset_index()

    # Pivot: Spalten pro use_tsm
    out_rows = []
    for case in sorted(g["_case"].unique()):
        g_case = g[g["_case"] == case]
        row = {"_case": case}

        # sicherstellen, dass beide vorhanden sind
        if not (True in g_case["use_tsm"].values and False in g_case["use_tsm"].values):
            # unvollständig
            out_rows.append(row)
            continue

        tsm = g_case[g_case["use_tsm"] == True].iloc[0]
        no  = g_case[g_case["use_tsm"] == False].iloc[0]

        for m in metric_cols:
            row[f"{m}_mean_noTSM"] = float(no[m])
            row[f"{m}_mean_TSM"] = float(tsm[m])
            row[f"{m}_gain_TSM_minus_noTSM"] = float(tsm[m] - no[m])

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs",
                    help="Ordner mit with_TSM_res und without_TSM_res")
    ap.add_argument("--with_dir", type=str, default="with_TSM_res")
    ap.add_argument("--without_dir", type=str, default="without_TSM_res")
    ap.add_argument("--out", type=str, default="runs/summary",
                    help="Output-Ordner für Summary-Dateien")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    with_root = runs_root / args.with_dir
    without_root = runs_root / args.without_dir

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_with = collect_runs(with_root)
    df_without = collect_runs(without_root)

    # Beide zusammenführen (einfach concat)
    df = pd.concat([df_with, df_without], ignore_index=True)

    if df.empty:
        print("Keine test_metrics.json gefunden. Check deine runs/* Struktur.")
        return

    # Sicherstellen, dass use_tsm boolean ist
    if "use_tsm" in df.columns:
        df["use_tsm"] = df["use_tsm"].astype(bool)

    # Welche Metriken interessieren uns?
    metric_cols = [
        "test_loss",
        "test_acc_motion",
        "test_acc_rot",
        "test_acc_shape",
        "test_acc_exact",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    # 1) Alles als CSV dumpen
    all_csv = out_dir / "all_runs.csv"
    df.to_csv(all_csv, index=False)
    print("Wrote:", all_csv)

    # 2) Mean/Std pro Case + use_tsm (über Seeds)
    stats = mean_std_table(df, group_cols=["_case", "use_tsm"], metric_cols=metric_cols)
    stats_csv = out_dir / "mean_std_by_case_and_tsm.csv"
    stats.to_csv(stats_csv, index=False)
    print("Wrote:", stats_csv)

    # 3) TSM Gain pro Case (TSM - noTSM) auf Mean-Werten
    gain = compute_tsm_gain(df, metric_cols=metric_cols)
    gain_csv = out_dir / "tsm_gain_by_case.csv"
    gain.to_csv(gain_csv, index=False)
    print("Wrote:", gain_csv)

    # 4) Bonus: je Case separate CSVs
    for case in sorted(df["_case"].unique()):
        df_case = df[df["_case"] == case].copy()
        p = out_dir / f"{case}_runs.csv"
        df_case.to_csv(p, index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
