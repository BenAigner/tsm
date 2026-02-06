import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scripts.CIFAR.cifar_sequence_dataset import CIFARSequenceDataset
from models.baseline_seq_cnn import BaselineSeqCNN
from models.tsm_cnn_old import TSM_CNN

def train_one(model, loader, device, epochs=2, lr=1e-3):
    model = model.to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        print(f"Epoch {epoch+1}: loss={total_loss/total:.4f} acc={correct/total:.4f}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    T = 4
    batch_size = 32
    epochs = 2

    transform = transforms.ToTensor()
    base_train = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    seq_train = CIFARSequenceDataset(base_train, T=T)

    loader = DataLoader(seq_train, batch_size=batch_size, shuffle=True, num_workers=0)

    print("\n=== Baseline (ohne TSM) ===")
    train_one(BaselineSeqCNN(num_classes=10, n_segment=T), loader, device, epochs=epochs)

    print("\n=== TSM_CNN (mit TSM) ===")
    train_one(TSM_CNN(num_classes=10, n_segment=T), loader, device, epochs=epochs)

if __name__ == "__main__":
    main()

# python -m scripts.train_compare_baseline_vs_tsm
