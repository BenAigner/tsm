import torch
import torch.nn as nn
from models.simple_cnn import SimpleCNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2
)


device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch+1}: Loss={total_loss:.3f}, Acc={correct/total:.3f}")

torch.save(model.state_dict(), "results/simple_cnn.pt")


# python -m scripts.train_simple_cnn