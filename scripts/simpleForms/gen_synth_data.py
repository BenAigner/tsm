import os
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Konstanten für Trans- und Rotationsgeschwindigkeit
FIXED_SPEED = 4.0  # px/frame für Bewegung
FIXED_OMEGA = 6.0  # Grad/frame für Rotation

# ----------------------------
# Labeldefinitionen
# ----------------------------
# Motion: fünf Klassen inkl. "keine Bewegung"
MOTION_LABELS = ["none", "left", "right", "up", "down"]
MOTION_TO_ID = {name: i for i, name in enumerate(MOTION_LABELS)}

# Rotation: ccw, cw, none
ROT_LABELS = ["ccw", "cw", "none"]
ROT_TO_ID = {name: i for i, name in enumerate(ROT_LABELS)}

# Form: Rechteck oder Dreieck
SHAPE_LABELS = ["rect", "tri"]
SHAPE_TO_ID ={name: i for i, name in enumerate(SHAPE_LABELS)} # rect = 0, tri = 1


@dataclass
class ClipParams:
    # Labels
    motion_name: str
    motion_id: int
    rot_name: str
    rot_id: int
    shape_id: int

    # Rendering
    shape: str          # "rect" | "tri"
    H: int
    W: int
    T: int
    radius: int         # grobe Größe (für rect/tri als Maß)
    thickness: int      # -1 = gefüllt

    # Bewegung
    x0: int
    y0: int
    vx: float           # px/frame
    vy: float           # px/frame

    # Rotation
    angle0: float       # Grad
    omega_deg_per_frame: float

    # leichte Störungen
    noise_std: float
    blur_ksize: int
    bg_level: int


def draw_shape(canvas: np.ndarray,
               shape: str,
               center: Tuple[int, int],
               radius: int,
               angle_deg: float,
               color: Tuple[int, int, int],
               thickness: int) -> None:
    """
    Zeichnet eine gefüllte Form (Rechteck oder Dreieck) auf den Canvas.
    Rotation wird über rotierte Polygonpunkte realisiert.
    """
    cx, cy = center

    if shape == "rect":
        # Rechteckpunkte im lokalen Koordinatensystem
        half_w = radius
        half_h = int(radius * 0.7)
        pts = np.array([
            [-half_w, -half_h],
            [ half_w, -half_h],
            [ half_w,  half_h],
            [-half_w,  half_h],
        ], dtype=np.float32)

    elif shape == "tri":
        # Dreieckpunkte im lokalen Koordinatensystem
        h = int(radius * 1.2)
        pts = np.array([
            [0, -h],
            [-radius, int(h * 0.6)],
            [ radius, int(h * 0.6)],
        ], dtype=np.float32)

    else:
        raise ValueError(f"Unknown shape: {shape}")

    # Rotation der Punkte um den Ursprung, dann Translation zum Zentrum
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
    """
    Rendert einen Clip als uint8-Array der Form [T, H, W, 3].
    """
    frames = np.empty((p.T, p.H, p.W, 3), dtype=np.uint8)

    x = float(p.x0)
    y = float(p.y0)
    angle = float(p.angle0)

    color = (0, 0, 0)  # Formfarbe: schwarz

    for t in range(p.T):
        # Hintergrund: leicht variierendes Weiß
        img = np.full((p.H, p.W, 3), p.bg_level, dtype=np.uint8)

        # Form zeichnen
        cx = int(round(x))
        cy = int(round(y))
        draw_shape(img, p.shape, (cx, cy), p.radius, angle, color, p.thickness)

        # additive Rauschkomponente (klein gehalten)
        if p.noise_std > 0:
            f = img.astype(np.float32)
            noise = rng.normal(0.0, p.noise_std, size=f.shape).astype(np.float32)
            f = np.clip(f + noise, 0, 255)
            img = f.astype(np.uint8)

        # optionaler leichter Blur
        if p.blur_ksize > 0:
            k = p.blur_ksize
            if k % 2 == 0:
                k += 1
            img = cv2.GaussianBlur(img, (k, k), 0)

        frames[t] = img

        # Zustand fortschreiben
        x += p.vx
        y += p.vy
        angle += p.omega_deg_per_frame

    return frames


def sample_params(H: int,
                  W: int,
                  T: int,
                  rng: np.random.Generator,
                  shape: str) -> ClipParams:
    """
    Samplet zufällige Clip-Parameter.
    Labels (Motion/Rotation) werden balanciert über einen äußeren Zyklus erzeugt.
    """
    # Platzhalter, Labels werden später gesetzt
    motion_name = "none"
    motion_id = MOTION_TO_ID[motion_name]
    rot_name = "none"
    rot_id = ROT_TO_ID[rot_name]

    shape_id = SHAPE_TO_ID[shape] # 0 für rect, 1 für tri


    # Objektgröße
    radius = int(rng.integers(low=max(8, min(H, W)//16), high=max(14, min(H, W)//8)))
    thickness = -1

    # Startwinkel
    angle0 = float(rng.uniform(0, 360))

    # Störungen (klein)
    bg_level = int(rng.integers(235, 256))
    noise_std = float(rng.uniform(0.0, 6.0))
    blur_ksize = int(rng.choice([0, 0, 0, 3]))

    # Bewegung/Rotation werden durch Labels gesetzt
    return ClipParams(
        motion_name=motion_name,
        motion_id=motion_id,
        rot_name=rot_name,
        rot_id=rot_id,
        shape=shape,
        shape_id=shape_id,
        H=H, W=W, T=T,
        radius=radius,
        thickness=thickness,
        x0=0, y0=0,
        vx=0.0, vy=0.0,
        angle0=angle0,
        omega_deg_per_frame=0.0,
        noise_std=noise_std,
        blur_ksize=blur_ksize,
        bg_level=bg_level,
    )


def set_motion(p: ClipParams, motion_name: str, rng: np.random.Generator) -> None:
    """
    Setzt Geschwindigkeit und Startposition so, dass die Form im Bild bleibt.
    """
    p.motion_name = motion_name
    p.motion_id = MOTION_TO_ID[motion_name]

    if motion_name == "none":
        vx, vy = 0.0, 0.0
        dx, dy = 0.0, 0.0
    else:
        speed = FIXED_SPEED  # px/frame
        vx, vy = 0.0, 0.0
        if motion_name == "left":
            vx = -speed
        elif motion_name == "right":
            vx = speed
        elif motion_name == "up":
            vy = -speed
        elif motion_name == "down":
            vy = speed

        dx = abs(vx) * (p.T - 1)
        dy = abs(vy) * (p.T - 1)

    p.vx, p.vy = vx, vy

    # Startposition so wählen, dass der Pfad innerhalb des Bildes liegt
    margin = p.radius + 3

    x_min = margin + int(dx if vx < 0 else 0)
    x_max = (p.W - margin) - int(dx if vx > 0 else 0)
    y_min = margin + int(dy if vy < 0 else 0)
    y_max = (p.H - margin) - int(dy if vy > 0 else 0)

    # robuste Fallbacks, falls Parameter selten zu eng werden
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


def set_rotation(p: ClipParams, rot_name: str, rng: np.random.Generator) -> None:
    """
    Setzt Rotationsgeschwindigkeit in Grad/Frame.
    ccw: positive Winkelgeschwindigkeit
    cw:  negative Winkelgeschwindigkeit
    none: 0
    """
    p.rot_name = rot_name
    p.rot_id = ROT_TO_ID[rot_name]

    if rot_name == "none":
        p.omega_deg_per_frame = 0.0
    else:
        # Magnitude bewusst klein bis moderat halten, damit pro Clip eine erkennbare Änderung entsteht
        mag = FIXED_OMEGA  # deg/frame
        p.omega_deg_per_frame = mag if rot_name == "ccw" else -mag


def save_preview_mp4(frames: np.ndarray, out_path: str, fps: int = 12) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    T, H, W, _ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    for t in range(T):
        vw.write(frames[t])
    vw.release()


def make_balanced_triples(n: int, rng: np.random.Generator) -> List[Tuple[str, str, str]]:
    """
    Erzeugt eine Liste von (motion, rot, shape) Triplen.
    Ziel: gleichmäßige Abdeckung über die Kartesischen Kombinationen.
    """
    combos = [(m, r, s) for m in MOTION_LABELS for r in ROT_LABELS for s in SHAPE_LABELS]
    reps = (n + len(combos) - 1) // len(combos)
    triples = (combos * reps)[:n]
    rng.shuffle(triples)
    return triples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="synth_multilabel", help="Output-Verzeichnis")
    ap.add_argument("--train", type=int, default=3000, help="Anzahl Train-Clips")
    ap.add_argument("--val", type=int, default=600, help="Anzahl Val-Clips")
    ap.add_argument("--test", type=int, default=600, help="Anzahl Test-Clips")
    ap.add_argument("--H", type=int, default=112)
    ap.add_argument("--W", type=int, default=112)
    ap.add_argument("--T", type=int, default=16, help="Frames pro Clip (entspricht n_segment)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--shapes", type=str, default="rect,tri", help="Komma-separiert: rect,tri")
    ap.add_argument("--preview", type=int, default=12, help="Anzahl Preview-mp4 pro Split")
    args = ap.parse_args()

    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    for s in shapes:
        if s not in ["rect", "tri"]:
            raise ValueError(f"Invalid shape: {s}")

    out_root = args.out
    os.makedirs(out_root, exist_ok=True)

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

        # Motion/Rotation Kombinationen balanciert
        triples = make_balanced_triples(n, rng)

        rows: List[Dict[str, Any]] = []
        pbar = tqdm(range(n), desc=f"Generating {split_name}")
        for i in pbar:
            motion_name, rot_name, shape_name = triples[i]

            p = sample_params(args.H, args.W, args.T, rng, shape_name)
            set_motion(p, motion_name, rng)
            set_rotation(p, rot_name, rng)

            frames = render_clip(p, rng)

            clip_path = os.path.join(clip_dir, f"{i:06d}.npz")
            np.savez_compressed(
                clip_path,
                frames=frames,            # [T,H,W,3] uint8
                motion=np.int64(p.motion_id),
                rot=np.int64(p.rot_id),
                shape=SHAPE_TO_ID[p.shape],
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


if __name__ == "__main__":
    main()
