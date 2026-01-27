import torch
from torch.utils.data import Dataset

class CIFARSequenceDataset(Dataset):

    """
        A custom Dataset class that generates sequences of CIFAR-10 images for temporal models.
        Label: Label of the last image in the sequence.
    """

    def __init__(self, base_dataset, T:int):
        self.base = base_dataset
        self.T = T

    def __len__(self):
        return len(self.base) - self.T + 1
    
    def __getitem__(self, idx):
        frames = []
        label = None
        for t in range(self.T):
            img, lbl = self.base[idx + t]
            frames.append(img)
            label = lbl  # Label of the last frame in the sequence
        frames = torch.stack(frames, dim=0)  # Shape: [T, C, H, W]
        return frames, label

