"""
synth_dataset_regression.py


Erwartete Dataset-Struktur:
  root/
    dataset_config.json          (optional, empfohlen)
    train/clips/*.npz
    val/clips/*.npz
    test/clips/*.npz

Jede .npz enthält:
  frames: uint8 [T, H, W, 3]
  motion: int64 scalar
  rot   : int64 scalar
  shape : int64 scalar
  speed     : float32 scalar   (px/frame, Betrag)
  omega_mag : float32 scalar   (deg/frame, Betrag)

Rückgabe:
  clip  : FloatTensor [T, C, H, W] in [0,1]
  motion: LongTensor  []
  rot   : LongTensor  []
  shape : LongTensor  []
  speed : FloatTensor []
  omega : FloatTensor []
"""
import os
import json
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SynthMultiLabelNPZRegression(Dataset):
    """
    Dataset für Multi-Task:
      - Klassifikation: motion, rot, shape
      - Regression: speed, omega_mag

    Optional kann speed/omega_mag normalisiert werden:
      speed_norm = speed / speed_max
      omega_norm = omega_mag / omega_max

    speed_max/omega_max werden bevorzugt aus dataset_config.json geladen.
    """

    def __init__(
        self,
        root: str,
        split: str,
        normalize_regression: bool = False,
    ):
        self.root = root
        self.split = split
        self.normalize_regression = bool(normalize_regression)

        self.clip_dir = os.path.join(root, split, "clips")
        if not os.path.isdir(self.clip_dir):
            raise FileNotFoundError(f"Clip-Ordner nicht gefunden: {self.clip_dir}")

        self.files = sorted([f for f in os.listdir(self.clip_dir) if f.endswith(".npz")])
        if not self.files:
            raise FileNotFoundError(f"Keine .npz Clips gefunden in: {self.clip_dir}")

        # Optional: Config laden (für Normalisierung)
        cfg_path = os.path.join(root, "dataset_config.json")
        self.cfg = self._load_json(cfg_path)

        # Maxima für Normalisierung bestimmen
        self.speed_max, self.omega_max = self._infer_norm_maxima(self.cfg)

        if self.normalize_regression:
            if self.speed_max is None or self.omega_max is None:
                raise ValueError(
                    "normalize_regression=True, aber speed_max/omega_max fehlen.\n"
                    "-> Stelle sicher, dass dataset_config.json speed_max und omega_max enthält\n"
                    "   (Generator: --store_norm_info)."
                )
            if self.speed_max <= 0 or self.omega_max <= 0:
                raise ValueError(f"Ungültige Maxima: speed_max={self.speed_max}, omega_max={self.omega_max}")

    @staticmethod
    def _load_json(path: str) -> Optional[Dict[str, Any]]:
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _infer_norm_maxima(cfg: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
        """
        Bestimmt speed_max/omega_max:

        Priorität:
          1) cfg["speed_max"], cfg["omega_max"]
          2) Fallback: max aus cfg["speed_range"] / cfg["omega_range"]
        """
        if not cfg:
            return None, None

        speed_max = cfg.get("speed_max", None)
        omega_max = cfg.get("omega_max", None)

        if speed_max is None:
            sr = cfg.get("speed_range", None)
            if isinstance(sr, list) and len(sr) == 2:
                speed_max = float(max(sr[0], sr[1]))

        if omega_max is None:
            orng = cfg.get("omega_range", None)
            if isinstance(orng, list) and len(orng) == 2:
                omega_max = float(max(orng[0], orng[1]))

        try:
            speed_max = float(speed_max) if speed_max is not None else None
        except Exception:
            speed_max = None

        try:
            omega_max = float(omega_max) if omega_max is not None else None
        except Exception:
            omega_max = None

        return speed_max, omega_max

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = os.path.join(self.clip_dir, self.files[idx])

        data = np.load(path, mmap_mode="r")
        # data = np.load(path)  # kein mmap, da wir kleine Clips haben und sofort alles brauchen

        # Pflichtfelder
        frames = data["frames"]  # uint8 [T,H,W,3]
        motion = int(data["motion"])
        rot = int(data["rot"])
        shape = int(data["shape"])

        # Regressionfelder (müssen existieren in regression-ready datasets)
        if "speed" not in data or "omega_mag" not in data:
            raise KeyError(
                f"NPZ enthält keine Regression-Keys ('speed', 'omega_mag'): {path}\n"
                "-> Stelle sicher, dass du die Regression-Datasets generiert hast."
            )

        speed = float(data["speed"])
        omega_mag = float(data["omega_mag"])

        # Frames: [T,H,W,3] -> [T,3,H,W], float in [0,1]
        clip = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        motion_t = torch.tensor(motion, dtype=torch.long)
        rot_t = torch.tensor(rot, dtype=torch.long)
        shape_t = torch.tensor(shape, dtype=torch.long)

        if self.normalize_regression:
            speed = speed / self.speed_max
            omega_mag = omega_mag / self.omega_max

        speed_t = torch.tensor(speed, dtype=torch.float32)
        omega_t = torch.tensor(omega_mag, dtype=torch.float32)

        return clip, motion_t, rot_t, shape_t, speed_t, omega_t
