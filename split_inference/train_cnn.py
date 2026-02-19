import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Allow running both:
# - as a module:  python -m split_inference.train_cnn
# - as a script:  python split_inference/train_cnn.py
if __package__:
    from .cnn_model import BigCNN, BiggerCNN, DeepCNN, HugeCNN, MiniResNet, SimpleCNN
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from split_inference.cnn_model import BigCNN, BiggerCNN, DeepCNN, HugeCNN, MiniResNet, SimpleCNN

try:
    from torchvision import datasets, transforms  # type: ignore

    _HAS_TORCHVISION = True
except ModuleNotFoundError:
    _HAS_TORCHVISION = False
    if __package__:
        from .mnist_fallback import MNISTDataset, mnist_tensor_transform
    else:
        from split_inference.mnist_fallback import MNISTDataset, mnist_tensor_transform

def train_model(model, train_loader, epochs=1, name="model"):
    print(f"\n--- Training {name} ---")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if i % 100 == 99:
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.4f}, Acc: {100 * correct / total:.2f}%")
                running_loss = 0.0
                
    # Save
    if not os.path.exists('split_inference'):
        os.makedirs('split_inference')
    
    save_path = f'split_inference/mnist_{name.lower()}.pth'
    torch.save(model.state_dict(), save_path)
    print(f"{name} saved to {save_path}")

def main():
    BATCH_SIZE = 64
    EPOCHS = 1 # Quick training for demo
    
    # Setup Data
    if _HAS_TORCHVISION:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    else:
        transform = mnist_tensor_transform
    
    print("Loading MNIST...")
    if _HAS_TORCHVISION:
        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    else:
        train_dataset = MNISTDataset(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Train Models
    models = [
        (SimpleCNN(), "SimpleCNN"),
        (DeepCNN(), "DeepCNN"),
        (MiniResNet(), "MiniResNet"),
        (BigCNN(), "BigCNN"),
        (BiggerCNN(), "BiggerCNN"),
        (HugeCNN(), "HugeCNN"),
    ]
    
    for model, name in models:
        train_model(model, train_loader, epochs=EPOCHS, name=name)

if __name__ == "__main__":
    main()
