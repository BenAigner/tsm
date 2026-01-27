from pathlib import Path
import torch
from torch.utils.data import DataLoader

from scripts.dogs_vs_cats.video_dataset import VideoFrameDataset

def main():
    project_root = Path(__file__).resolve().parents[2]
    frames_root = project_root / "data" / "frames"

    ds = VideoFrameDataset(frames_root=frames_root, T=8, split="train", samples_per_video=2)
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    x, y = next(iter(dl))
    print("x shape:", x.shape)  # erwartet: [N, T, C, H, W]
    print("y shape:", y.shape)  # erwartet: [N]
    print("labels:", y.tolist())
    print("class_to_idx:", ds.class_to_idx)

if __name__ == "__main__":
    main()
