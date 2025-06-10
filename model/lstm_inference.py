import torch
import torch.nn.functional as F
import pickle
import numpy as np
import mlflow.pytorch
import os
from tokenizer import Tokenizer
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / 'secrets_db.env')

HOST = os.getenv('HOST')
TOKEN = os.getenv('TOKEN')
os.environ["DATABRICKS_HOST"] = HOST
os.environ["DATABRICKS_TOKEN"] = TOKEN

class NewsClassifier:
    def __init__(self, mlflow_model_uri, categories, maxlen=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.categories = categories
        self.maxlen = maxlen

        # Get the local path to the MLflow model artifacts
        local_model_path = mlflow.artifacts.download_artifacts(mlflow_model_uri)
        print(f"Model artifacts downloaded to: {local_model_path}")

        model = torch.jit.load(os.path.join(local_model_path, "best_model.pt"), map_location=self.device)
        self.model = model.to(self.device)
        self.model.eval()

        # Load tokenizer from the artifact directory
        tokenizer_path = os.path.join(local_model_path, "tokenizer.json")
        self.tokenizer = Tokenizer.load(tokenizer_path)

        self.vocab_size = self.tokenizer.vocab_size() + 1

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        sequences = self.tokenizer.texts_to_sequences(texts)
        sequences = self.tokenizer.pad_sequences(sequences)
        sequences = np.clip(sequences, 0, self.vocab_size - 1)
        inputs = torch.tensor(sequences).long().to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)
            probs = F.softmax(outputs, dim=1)
            pred_classes = torch.argmax(probs, dim=1) + 1  # 1-based indexing

        return [
            (pred.item(), prob.cpu().numpy())
            for pred, prob in zip(pred_classes, probs)
        ]

    def predict_label(self, texts):
        results = self.predict(texts)
        return [
            (self.categories[pred - 1], confidence_scores)
            for pred, confidence_scores in results
        ]

test_text = "The stock market is experiencing significant fluctuations due to global economic changes."
categories = ['World News', 'Sports News', 'Business News', 'Science-Technology News']

def default_classifier():
    # mlflow.set_tracking_uri('http://nowhere.cn:5000')
    mlflow.set_tracking_uri('databricks')
    mlflow.set_registry_uri('databricks-uc')

    # mlflow_model_uri = "models:/Classifier@Champion" 
    mlflow_model_uri = "models:/workspace.default.Classifier@Champion" 
    return NewsClassifier(mlflow_model_uri, categories)

if __name__ == "__main__":
    results = default_classifier().predict_label(test_text)

    for label, confidence in results:
        print(f"Predicted Class Label: {label}")
        print(f"Confidence Scores: {confidence}")

