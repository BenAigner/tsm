#!/usr/bin/env python3
"""
gen_all_reg_datasets.py

Erzeugt automatisch alle 12 Regression-Datensätze:

Seeds:
  - 123
  - 456
  - 789

Cases:
  - FIX_MOT_FIX_ROT
  - FIX_MOT_VAR_ROT
  - VAR_MOT_FIX_ROT
  - VAR_MOT_VAR_ROT

Output Root:
  data/simpleForms_Regression/
"""

import subprocess
from pathlib import Path

# ==============================
# Konfiguration
# ==============================

SEEDS = [123, 456, 789]

CASES = [
    "FIX_MOT_FIX_ROT",
    "FIX_MOT_VAR_ROT",
    "VAR_MOT_FIX_ROT",
    "VAR_MOT_VAR_ROT",
]

# ✅ Genau dein gewünschter Output-Ordner
BASE_OUT = Path("data/simpleForms_Regression")

# ✅ Pfad zu deinem Generator-Skript (so wie im Screenshot)
GEN_SCRIPT = Path("scripts/simpleForms_Regression/gen_synth_data_regression.py")

# Regression-Intervalle (kontinuierlich)
SPEED_RANGE = "0,10"   # px/frame
OMEGA_RANGE = "0,10"   # deg/frame

# Fixed Werte (für fixed cases)
FIXED_SPEED = 4.0
FIXED_OMEGA = 6.0


def get_modes(case: str) -> tuple[str, str]:
    """Mapping Case → (motion_mode, rot_mode)."""
    if case == "FIX_MOT_FIX_ROT":
        return "fixed", "fixed"
    if case == "FIX_MOT_VAR_ROT":
        return "fixed", "variable"
    if case == "VAR_MOT_FIX_ROT":
        return "variable", "fixed"
    if case == "VAR_MOT_VAR_ROT":
        return "variable", "variable"
    raise ValueError(f"Unknown case: {case}")


def run_one(case: str, seed: int) -> None:
    motion_mode, rot_mode = get_modes(case)

    out_dir = BASE_OUT / f"{case}_s{seed}"

    cmd = [
        "python", str(GEN_SCRIPT),
        "--out", str(out_dir),
        "--on_exists", "overwrite",          # überschreibt gezielt nur den dataset-ordner
        "--train", "6000",
        "--val", "1200",
        "--test", "1200",
        "--motion_mode", motion_mode,
        "--rot_mode", rot_mode,
        "--speed_range", SPEED_RANGE,
        "--omega_range", OMEGA_RANGE,
        "--fixed_speed", str(FIXED_SPEED),
        "--fixed_omega", str(FIXED_OMEGA),
        "--seed", str(seed),
        "--store_norm_info",                # schreibt speed_max/omega_max in dataset_config.json
    ]

    print("\n" + "=" * 90)
    print(f"Generating dataset: {case} | seed={seed}")
    print(f"Output: {out_dir}")
    print("CMD:", " ".join(cmd))
    print("=" * 90)

    subprocess.run(cmd, check=True)


def main() -> None:
    if not GEN_SCRIPT.exists():
        raise FileNotFoundError(
            f"Generator-Skript nicht gefunden: {GEN_SCRIPT}\n"
            f"Prüfe Pfad/Name. (Laut Screenshot sollte es genau so heißen.)"
        )

    BASE_OUT.mkdir(parents=True, exist_ok=True)

    for case in CASES:
        for seed in SEEDS:
            run_one(case, seed)

    print("\n✅ Done. Alle 12 Regression-Datasets wurden erzeugt in:")
    print(f"   {BASE_OUT.resolve()}")


if __name__ == "__main__":
    main()
