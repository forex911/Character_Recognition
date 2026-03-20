import torch
import numpy as np
import torch.nn.functional as F

from model.cnn import CharacterCNN
from utils.config import MODEL_PATH
from utils.label_map import index_to_char


class Predictor:
    def __init__(self):
        self.model = CharacterCNN()
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu")
        )
        self.model.eval()

        # Debug check (safe to keep)
        weight_sum = sum(p.sum().item() for p in self.model.parameters())
        print("✅ Model weights sum:", weight_sum)

    def predict_with_confidence(self, image: np.ndarray):
        """
        Returns:
            (prediction_char, confidence_float)
        """
        tensor = torch.from_numpy(image)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)

        return index_to_char(idx.item()), conf.item()

    def predict(self, image: np.ndarray, threshold: float = 0.6) -> str:
        """
        Simple prediction (used if confidence not needed)
        """
        pred, conf = self.predict_with_confidence(image)

        if conf < threshold:
            return "?"

        return pred
