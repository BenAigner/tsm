import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SynthMultiLabelNPZ(Dataset):
    """
    Lädt Clips aus einem Dataset-Root, das so aufgebaut ist:

      root/
        train/clips/*.npz
        val/clips/*.npz
        test/clips/*.npz

    Jede .npz enthält:
      frames: uint8 [T, H, W, 3]
      motion: int64 scalar
      rot: int64 scalar
      shape: int64 scalar
      (optional) speed, omega_mag

    Rückgabe:
      clip  : FloatTensor [T, C, H, W] in [0,1]
      motion: LongTensor  []  (0..4)
      rot   : LongTensor  []  (0..2)
      shape : LongTensor  []  (0..1)
    """
    def __init__(self, root: str, split: str):
        self.root = root
        self.split = split
        self.clip_dir = os.path.join(root, split, "clips")
        if not os.path.isdir(self.clip_dir):
            raise FileNotFoundError(f"Clip-Ordner nicht gefunden: {self.clip_dir}")

        self.files = sorted([f for f in os.listdir(self.clip_dir) if f.endswith(".npz")])
        if not self.files:
            raise FileNotFoundError(f"Keine .npz Clips gefunden in: {self.clip_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.clip_dir, self.files[idx])

        # mmap_mode spart RAM, weil frames nicht komplett kopiert werden müssen
        data = np.load(path, mmap_mode="r")

        frames = data["frames"]              # uint8 [T,H,W,3]
        motion = int(data["motion"])
        rot = int(data["rot"])
        shape = int(data["shape"])

        clip = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        return (
            clip,
            torch.tensor(motion, dtype=torch.long),
            torch.tensor(rot, dtype=torch.long),
            torch.tensor(shape, dtype=torch.long),
        )
