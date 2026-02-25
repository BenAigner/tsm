#!/usr/bin/env python3
"""
gen_synth_data_regression.py

Erzeugt synthetische Clip-Datasets (train/val/test) wie dein bisheriges Skript,
aber mit kontinuierlichen (Regression) Targets für:

- speed        : Translationsgeschwindigkeit in px/frame (Betrag)
- omega_mag    : Rotationsgeschwindigkeit in deg/frame (Betrag)

Die Richtungen (motion/rot) bleiben weiterhin Klassifikation:
- motion: none/left/right/up/down
- rot  : ccw/cw/none
- shape: rect/tri

Wichtig:
- Bei motion="none" wird speed automatisch 0 gesetzt.
- Bei rot="none" wird omega_mag automatisch 0 gesetzt.
- Bei rot="cw"/"ccw" wird omega_deg_per_frame entsprechend +/-omega_mag gesetzt.

Dataset-Struktur:
  out_root/
    dataset_config.json
    labels_motion.csv
    labels_rot.csv
    labels_shape.csv
    train/clips/*.npz
    train/meta.csv
    train/preview/*.mp4
    val/...
    test/...

Jede .npz enthält:
  frames: uint8 [T,H,W,3]
  motion: int64 scalar
  rot   : int64 scalar
  shape : int64 scalar
  speed : float32 scalar
  omega_mag: float32 scalar
"""

import os
import math
import json
import argparse
import shutil
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ----------------------------
# Labels
# ----------------------------
MOTION_LABELS = ["none", "left", "right", "up", "down"]
MOTION_TO_ID = {name: i for i, name in enumerate(MOTION_LABELS)}

ROT_LABELS = ["ccw", "cw", "none"]
ROT_TO_ID = {name: i for i, name in enumerate(ROT_LABELS)}

SHAPE_LABELS = ["rect", "tri"]
SHAPE_TO_ID = {name: i for i, name in enumerate(SHAPE_LABELS)}  # rect=0, tri=1


@dataclass
class ClipParams:
    # Labels
    motion_name: str
    motion_id: int
    rot_name: str
    rot_id: int
    shape_id: int

    # Rendering
    shape: str
    H: int
    W: int
    T: int
    radius: int
    thickness: int

    # Bewegung
    speed: float              # px/frame (Betrag)
    x0: int
    y0: int
    vx: float
    vy: float

    # Rotation
    angle0: float
    omega_deg_per_frame: float  # signed deg/frame
    omega_mag: float            # Betrag deg/frame

    # leichte Störungen
    noise_std: float
    blur_ksize: int
    bg_level: int


def parse_range(s: str) -> Tuple[float, float]:
    """
    Parse "0,10" -> (0.0, 10.0)
    """
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Range muss exakt 2 Werte haben, z.B. '0,10'")
    a, b = float(parts[0]), float(parts[1])
    lo, hi = (a, b) if a <= b else (b, a)
    return lo, hi


def draw_shape(
    canvas: np.ndarray,
    shape: str,
    center: Tuple[int, int],
    radius: int,
    angle_deg: float,
    color: Tuple[int, int, int],
    thickness: int
) -> None:
    """
    Zeichnet eine Form (rect/tri) auf den Canvas.
    Rotation erfolgt über rotierte Polygonpunkte.
    """
    cx, cy = center

    if shape == "rect":
        half_w = radius
        half_h = int(radius * 0.7)
        pts = np.array([
            [-half_w, -half_h],
            [ half_w, -half_h],
            [ half_w,  half_h],
            [-half_w,  half_h],
        ], dtype=np.float32)

    elif shape == "tri":
        h = int(radius * 1.2)
        pts = np.array([
            [0, -h],
            [-radius, int(h * 0.6)],
            [ radius, int(h * 0.6)],
        ], dtype=np.float32)

    else:
        raise ValueError(f"Unknown shape: {shape}")

    theta = math.radians(angle_deg)
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta),  math.cos(theta)]], dtype=np.float32)

    pts_rot = (pts @ R.T) + np.array([cx, cy], dtype=np.float32)
    pts_int = np.round(pts_rot).astype(np.int32)

    if thickness == -1:
        cv2.fillPoly(canvas, [pts_int], color, lineType=cv2.LINE_AA)
    else:
        cv2.polylines(canvas, [pts_int], isClosed=True, color=color,
                      thickness=thickness, lineType=cv2.LINE_AA)


def render_clip(p: ClipParams, rng: np.random.Generator) -> np.ndarray:
    """Rendert einen Clip als uint8-Array [T, H, W, 3]."""
    frames = np.empty((p.T, p.H, p.W, 3), dtype=np.uint8)

    x = float(p.x0)
    y = float(p.y0)
    angle = float(p.angle0)

    color = (0, 0, 0)  # Formfarbe: schwarz

    for _t in range(p.T):
        img = np.full((p.H, p.W, 3), p.bg_level, dtype=np.uint8)

        cx = int(round(x))
        cy = int(round(y))
        draw_shape(img, p.shape, (cx, cy), p.radius, angle, color, p.thickness)

        # leichte Störungen (klein halten)
        if p.noise_std > 0:
            f = img.astype(np.float32)
            noise = rng.normal(0.0, p.noise_std, size=f.shape).astype(np.float32)
            f = np.clip(f + noise, 0, 255)
            img = f.astype(np.uint8)

        if p.blur_ksize > 0:
            k = p.blur_ksize + (1 - (p.blur_ksize % 2))  # make odd if needed
            img = cv2.GaussianBlur(img, (k, k), 0)

        frames[_t] = img

        # Zustand fortschreiben
        x += p.vx
        y += p.vy
        angle += p.omega_deg_per_frame

    return frames


def sample_base_params(H: int, W: int, T: int, rng: np.random.Generator, shape: str) -> ClipParams:
    """
    Samplet zufällige Renderparameter (Labels & speeds werden später gesetzt).
    """
    radius = int(rng.integers(low=max(8, min(H, W)//16), high=max(14, min(H, W)//8)))
    thickness = -1
    angle0 = float(rng.uniform(0, 360))

    bg_level = int(rng.integers(235, 256))
    noise_std = float(rng.uniform(0.0, 2.0))
    blur_ksize = int(rng.choice([0, 0, 0, 3]))

    return ClipParams(
        motion_name="none",
        motion_id=MOTION_TO_ID["none"],
        rot_name="none",
        rot_id=ROT_TO_ID["none"],
        shape_id=SHAPE_TO_ID[shape],
        shape=shape,
        H=H, W=W, T=T,
        radius=radius,
        thickness=thickness,
        speed=0.0,
        x0=0, y0=0,
        vx=0.0, vy=0.0,
        angle0=angle0,
        omega_deg_per_frame=0.0,
        omega_mag=0.0,
        noise_std=noise_std,
        blur_ksize=blur_ksize,
        bg_level=bg_level,
    )


def set_motion(
    p: ClipParams,
    motion_name: str,
    rng: np.random.Generator,
    motion_mode: str,
    fixed_speed: float,
    speed_range: Tuple[float, float],
) -> None:
    """
    Setzt Geschwindigkeit und Startposition so, dass die Form im Bild bleibt.

    motion_mode:
      - fixed    -> immer fixed_speed
      - variable -> speed ~ Uniform(speed_range)
    """
    p.motion_name = motion_name
    p.motion_id = MOTION_TO_ID[motion_name]

    if motion_name == "none":
        p.speed = 0.0
        p.vx, p.vy = 0.0, 0.0
        dx, dy = 0.0, 0.0
    else:
        if motion_mode == "fixed":
            speed = float(fixed_speed)
        elif motion_mode == "variable":
            lo, hi = speed_range
            speed = float(rng.uniform(lo, hi))
        else:
            raise ValueError(f"Unknown motion_mode: {motion_mode}")

        # Sicherheitsnetz: negative/NaN vermeiden
        if not np.isfinite(speed) or speed < 0:
            speed = 0.0

        p.speed = speed

        vx, vy = 0.0, 0.0
        if motion_name == "left":
            vx = -speed
        elif motion_name == "right":
            vx = speed
        elif motion_name == "up":
            vy = -speed
        elif motion_name == "down":
            vy = speed

        p.vx, p.vy = vx, vy
        dx = abs(vx) * (p.T - 1)
        dy = abs(vy) * (p.T - 1)

    # Startposition so wählen, dass der Pfad innerhalb des Bildes bleibt
    margin = p.radius + 3
    vx, vy = p.vx, p.vy

    x_min = margin + int(dx if vx < 0 else 0)
    x_max = (p.W - margin) - int(dx if vx > 0 else 0)
    y_min = margin + int(dy if vy < 0 else 0)
    y_max = (p.H - margin) - int(dy if vy > 0 else 0)

    # robuste Fallbacks
    x_min = min(max(x_min, margin), p.W - margin)
    x_max = max(min(x_max, p.W - margin), margin)
    y_min = min(max(y_min, margin), p.H - margin)
    y_max = max(min(y_max, p.H - margin), margin)

    if x_max < x_min:
        x_min, x_max = margin, p.W - margin
    if y_max < y_min:
        y_min, y_max = margin, p.H - margin

    p.x0 = int(rng.integers(x_min, x_max + 1))
    p.y0 = int(rng.integers(y_min, y_max + 1))


def set_rotation(
    p: ClipParams,
    rot_name: str,
    rng: np.random.Generator,
    rot_mode: str,
    fixed_omega: float,
    omega_range: Tuple[float, float],
) -> None:
    """
    Setzt Rotationsgeschwindigkeit in Grad/Frame.

    rot_mode:
      - fixed    -> immer fixed_omega (Betrag)
      - variable -> omega_mag ~ Uniform(omega_range)
    """
    p.rot_name = rot_name
    p.rot_id = ROT_TO_ID[rot_name]

    if rot_name == "none":
        p.omega_deg_per_frame = 0.0
        p.omega_mag = 0.0
        return

    if rot_mode == "fixed":
        mag = float(fixed_omega)
    elif rot_mode == "variable":
        lo, hi = omega_range
        mag = float(rng.uniform(lo, hi))
    else:
        raise ValueError(f"Unknown rot_mode: {rot_mode}")

    if not np.isfinite(mag) or mag < 0:
        mag = 0.0

    p.omega_mag = mag
    p.omega_deg_per_frame = mag if rot_name == "ccw" else -mag


def save_preview_mp4(frames: np.ndarray, out_path: str, fps: int = 12) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    T, H, W, _ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    for t in range(T):
        vw.write(frames[t])
    vw.release()


def make_balanced_triples(n: int, rng: np.random.Generator, shapes: List[str]) -> List[Tuple[str, str, str]]:
    """
    Erzeugt eine Liste von (motion, rot, shape) Tripeln.
    Ziel: gleichmäßige Abdeckung über alle Kombinationen.
    """
    combos = [(m, r, s) for m in MOTION_LABELS for r in ROT_LABELS for s in shapes]
    reps = (n + len(combos) - 1) // len(combos)
    triples = (combos * reps)[:n]
    rng.shuffle(triples)
    return triples


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--out", type=str, required=True, help="Output-Verzeichnis (Dataset-Root)")
    ap.add_argument("--on_exists", type=str, default="error", choices=["error", "overwrite"],
                    help="Wenn --out existiert: error (abbrechen) oder overwrite (löschen & neu schreiben)")

    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--val", type=int, default=1200)
    ap.add_argument("--test", type=int, default=1200)

    ap.add_argument("--H", type=int, default=112)
    ap.add_argument("--W", type=int, default=112)
    ap.add_argument("--T", type=int, default=16, help="Frames pro Clip (entspricht n_segment)")
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--shapes", type=str, default="rect,tri", help="Komma-separiert: rect,tri")
    ap.add_argument("--preview", type=int, default=12, help="Anzahl Preview-mp4 pro Split")

    # Modes wie vorher, nur variable ist jetzt kontinuierlich im Intervall
    ap.add_argument("--motion_mode", type=str, default="fixed", choices=["fixed", "variable"])
    ap.add_argument("--rot_mode", type=str, default="fixed", choices=["fixed", "variable"])

    # Fixed Werte
    ap.add_argument("--fixed_speed", type=float, default=4.0, help="px/frame (Betrag)")
    ap.add_argument("--fixed_omega", type=float, default=6.0, help="deg/frame (Betrag)")

    # Continuous ranges
    ap.add_argument("--speed_range", type=str, default="0,10", help="z.B. '0,10' px/frame")
    ap.add_argument("--omega_range", type=str, default="0,10", help="z.B. '0,10' deg/frame")

    # Optional: Normalisierung dokumentieren (Training kann später darauf zugreifen)
    ap.add_argument("--store_norm_info", action="store_true",
                    help="Schreibt zusätzlich speed_max/omega_max in dataset_config.json (hilft fürs Training).")

    args = ap.parse_args()

    out_root = args.out

    # Sicherheitsnetz: Datasets nicht still überschreiben
    if os.path.exists(out_root) and os.listdir(out_root):
        if args.on_exists == "error":
            raise FileExistsError(
                f"Dataset-Ordner existiert bereits und ist nicht leer: {out_root}\n"
                f"Wenn du ihn wirklich neu erzeugen willst: --on_exists overwrite"
            )
        else:
            print(f"[INFO] Lösche bestehenden Dataset-Ordner und erzeuge neu: {out_root}")
            shutil.rmtree(out_root)

    os.makedirs(out_root, exist_ok=True)

    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    for s in shapes:
        if s not in SHAPE_LABELS:
            raise ValueError(f"Invalid shape: {s}")

    speed_range = parse_range(args.speed_range)
    omega_range = parse_range(args.omega_range)

    # Config abspeichern (für Reproduzierbarkeit)
    cfg: Dict[str, Any] = {
        "H": args.H,
        "W": args.W,
        "T": args.T,
        "seed": args.seed,
        "shapes": shapes,
        "motion_mode": args.motion_mode,
        "rot_mode": args.rot_mode,
        "fixed_speed": float(args.fixed_speed),
        "fixed_omega": float(args.fixed_omega),
        "speed_range": [float(speed_range[0]), float(speed_range[1])],
        "omega_range": [float(omega_range[0]), float(omega_range[1])],
        "counts": {"train": args.train, "val": args.val, "test": args.test},
        "note": "Regression-ready: speed/omega_mag are continuous (uniform) when mode=variable.",
    }
    if args.store_norm_info:
        # Bei Regression ist es praktisch, Targets z.B. auf [0,1] zu normieren.
        # Dann kann später im Training norm = value / max_value gemacht werden.
        cfg["speed_max"] = float(max(speed_range))
        cfg["omega_max"] = float(max(omega_range))

    with open(os.path.join(out_root, "dataset_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    splits = [("train", args.train, args.seed),
              ("val", args.val, args.seed + 1),
              ("test", args.test, args.seed + 2)]

    for split_name, n, split_seed in splits:
        split_dir = os.path.join(out_root, split_name)
        clip_dir = os.path.join(split_dir, "clips")
        prev_dir = os.path.join(split_dir, "preview")
        os.makedirs(clip_dir, exist_ok=True)
        os.makedirs(prev_dir, exist_ok=True)

        rng = np.random.default_rng(split_seed)
        triples = make_balanced_triples(n, rng, shapes)

        rows: List[Dict[str, Any]] = []
        pbar = tqdm(range(n), desc=f"Generating {split_name}")

        for i in pbar:
            motion_name, rot_name, shape_name = triples[i]

            p = sample_base_params(args.H, args.W, args.T, rng, shape_name)
            set_motion(p, motion_name, rng, args.motion_mode, args.fixed_speed, speed_range)
            set_rotation(p, rot_name, rng, args.rot_mode, args.fixed_omega, omega_range)

            frames = render_clip(p, rng)

            clip_path = os.path.join(clip_dir, f"{i:06d}.npz")
            np.savez_compressed(
                clip_path,
                frames=frames,
                motion=np.int64(p.motion_id),
                rot=np.int64(p.rot_id),
                shape=np.int64(p.shape_id),
                speed=np.float32(p.speed),
                omega_mag=np.float32(p.omega_mag),
            )

            row = asdict(p)
            row["clip_path"] = os.path.relpath(clip_path, out_root)
            rows.append(row)

            if i < args.preview:
                mp4_path = os.path.join(prev_dir, f"{i:03d}_m-{motion_name}_r-{rot_name}_{p.shape}.mp4")
                save_preview_mp4(frames, mp4_path, fps=12)

        pd.DataFrame(rows).to_csv(os.path.join(split_dir, "meta.csv"), index=False)

    # Label-Mappings separat speichern
    pd.DataFrame([{"motion_id": i, "motion_name": n} for n, i in MOTION_TO_ID.items()]) \
        .sort_values("motion_id").to_csv(os.path.join(out_root, "labels_motion.csv"), index=False)

    pd.DataFrame([{"rot_id": i, "rot_name": n} for n, i in ROT_TO_ID.items()]) \
        .sort_values("rot_id").to_csv(os.path.join(out_root, "labels_rot.csv"), index=False)

    pd.DataFrame([{"shape_id": i, "shape_name": n} for n, i in SHAPE_TO_ID.items()]) \
        .sort_values("shape_id").to_csv(os.path.join(out_root, "labels_shape.csv"), index=False)

    print(f"\nDone. Dataset written to: {out_root}")
    print("Splits: train/val/test, each contains clips/*.npz, meta.csv, preview/*.mp4")
    print(f"Config: {os.path.join(out_root, 'dataset_config.json')}")


if __name__ == "__main__":
    main()
