import torch
from data.loader import get_dataloaders
from model.cnn import CharacterCNN
from utils.config import MODEL_PATH

def evaluate():
    _, test_loader = get_dataloaders()

    model = CharacterCNN()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            labels -= 1
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    evaluate()
