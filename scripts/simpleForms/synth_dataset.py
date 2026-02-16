import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SynthMultiLabelNPZ(Dataset):
    """
    Lädt Clips, die von make_synth_multilabel.py erzeugt wurden.

    Rückgabe:
      clip  : FloatTensor [T, C, H, W] in [0,1]
      motion: LongTensor  []  (0..4)
      rot   : LongTensor  []  (0..2)
    """
    def __init__(self, root: str, split: str):
        self.root = root
        self.split = split
        self.clip_dir = os.path.join(root, split, "clips")
        self.files = sorted([f for f in os.listdir(self.clip_dir) if f.endswith(".npz")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.clip_dir, self.files[idx])
        data = np.load(path, mmap_mode="r")
        frames = data["frames"]           # uint8 [T,H,W,3]
        motion = int(data["motion"])      # int64 scalar
        rot = int(data["rot"])            # int64 scalar
        shape = int(data["shape"])        # int64 scalar

        clip = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        return clip, torch.tensor(motion, dtype=torch.long), torch.tensor(rot, dtype=torch.long), torch.tensor(shape, dtype=torch.long) 
