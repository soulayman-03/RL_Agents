import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .cnn_model import SimpleCNN, DeepCNN, MiniResNet
import os

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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("Loading MNIST...")
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Train Models
    models = [
        (SimpleCNN(), "SimpleCNN"),
        (DeepCNN(), "DeepCNN"),
        (MiniResNet(), "MiniResNet")
    ]
    
    for model, name in models:
        train_model(model, train_loader, epochs=EPOCHS, name=name)

if __name__ == "__main__":
    main()
