import torch
from torch import nn
from data.loader import get_dataloaders
from model.cnn import CharacterCNN
from utils.config import MODEL_PATH, EPOCHS, LEARNING_RATE

def train():
    train_loader = get_dataloaders()

    model = CharacterCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("✅ Model saved:", MODEL_PATH)

if __name__ == "__main__":
    train()
