from pathlib import Path
from typing import List, Tuple, Dict
import random

import torch
from torch.utils.data import Dataset
from PIL import Image

from torchvision import transforms
import torchvision.transforms.functional as F


class VideoFrameDataset(Dataset):
    """
    Liest Frames aus data/frames/<class>/<video>/*.jpg und liefert Sequenzen fester Länge T.

    Ausgabe:
        x: [T, C, H, W]
        y: int
    """

    def __init__(
        self,
        frames_root: str | Path,
        T: int = 8,
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
        stride: int = 1,
        samples_per_video: int = 10,
        image_size: int = 128,
    ):
        super().__init__()
        self.frames_root = Path(frames_root)
        self.T = T
        self.stride = stride
        self.samples_per_video = samples_per_video

        assert split in {"train", "test"}
        self.split = split

        class_dirs = sorted([p for p in self.frames_root.iterdir() if p.is_dir()])
        if not class_dirs:
            raise RuntimeError(f"Keine Klassenordner in {self.frames_root} gefunden.")

        self.class_to_idx: Dict[str, int] = {d.name: i for i, d in enumerate(class_dirs)}

        videos_by_class: Dict[str, List[Path]] = {}
        for cdir in class_dirs:
            video_dirs = sorted([p for p in cdir.iterdir() if p.is_dir()])
            if not video_dirs:
                raise RuntimeError(f"Keine Video-Ordner in {cdir} gefunden.")
            videos_by_class[cdir.name] = video_dirs

        rng = random.Random(seed)
        self.video_items: List[Tuple[Path, int]] = []

        for cname, vdirs in videos_by_class.items():
            vdirs_copy = vdirs[:]
            rng.shuffle(vdirs_copy)

            n_train = max(1, int(round(len(vdirs_copy) * train_ratio)))
            if split == "train":
                chosen = vdirs_copy[:n_train]
            else:
                chosen = vdirs_copy[n_train:]

            label = self.class_to_idx[cname]
            for vd in chosen:
                self.video_items.append((vd, label))

        if not self.video_items:
            raise RuntimeError("Split ist leer. Prüfe train_ratio oder ob genug Videos vorhanden sind.")

        self.image_size = image_size

        # Train-Transform (wird clip-konsistent angewendet)
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        # Test-Transform (deterministisch)
        self.tf_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            self.normalize,
        ])

        # Parameter für ColorJitter (werden pro Clip einmal gesampelt)
        self.jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    def __len__(self) -> int:
        return len(self.video_items) * self.samples_per_video

    def _list_frames(self, video_dir: Path) -> List[Path]:
        return sorted([p for p in video_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}])

    def _choose_sequence(self, frames: List[Path], sample_idx_in_video: int) -> List[Path]:
        needed = 1 + (self.T - 1) * self.stride

        if len(frames) >= needed:
            max_start = len(frames) - needed

            if self.split == "train":
                start = random.randint(0, max_start)
            else:
                # deterministischer Start für Test (abhängig vom sample index)
                start = min(sample_idx_in_video, max_start)

            return [frames[start + i * self.stride] for i in range(self.T)]

        chosen = []
        for i in range(self.T):
            src_idx = min(i * self.stride, len(frames) - 1)
            chosen.append(frames[src_idx])
        return chosen

    def _transform_clip_train(self, pil_images):
        # RandomResizedCrop-Parameter einmal pro Clip sampeln
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            pil_images[0], scale=(0.7, 1.0), ratio=(0.8, 1.2)
        )

        # HorizontalFlip einmal pro Clip entscheiden
        do_flip = random.random() < 0.5

        # ColorJitter-Faktoren einmal pro Clip sampeln
        # Wertebereich entspricht ungefähr brightness/contrast/saturation=0.2
        b = 1.0 + random.uniform(-0.2, 0.2)
        c = 1.0 + random.uniform(-0.2, 0.2)
        s = 1.0 + random.uniform(-0.2, 0.2)

        out = []
        for img in pil_images:
            # gleicher Crop für alle Frames
            img = F.resized_crop(img, i, j, h, w, (self.image_size, self.image_size))

            # gleicher Flip für alle Frames
            if do_flip:
                img = F.hflip(img)

            # gleicher ColorJitter für alle Frames
            img = F.adjust_brightness(img, b)
            img = F.adjust_contrast(img, c)
            img = F.adjust_saturation(img, s)

            # zu Tensor + Normalisierung
            x = F.to_tensor(img)
            x = self.normalize(x)
            out.append(x)

        return torch.stack(out, dim=0)  # [T, C, H, W]

    def __getitem__(self, idx: int):
        video_index = idx // self.samples_per_video
        sample_idx_in_video = idx % self.samples_per_video

        video_dir, label = self.video_items[video_index]
        frames = self._list_frames(video_dir)
        if len(frames) == 0:
            raise RuntimeError(f"Keine Frames gefunden in {video_dir}")

        chosen = self._choose_sequence(frames, sample_idx_in_video)

        pil_imgs = [Image.open(fp).convert("RGB") for fp in chosen]

        if self.split == "train":
            x = self._transform_clip_train(pil_imgs)
        else:
            x = torch.stack([self.tf_test(im) for im in pil_imgs], dim=0)

        return x, label
